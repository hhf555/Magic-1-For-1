import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../')));

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, '../uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueName = `${uuidv4()}_${file.originalname}`;
        cb(null, uniqueName);
    }
});

const upload = multer({ 
    storage,
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    },
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'));
        }
    }
});

// Store active WebSocket connections
const clients = new Set();

// WebSocket connection handling
wss.on('connection', (ws) => {
    clients.add(ws);
    console.log('Client connected. Total clients:', clients.size);
    
    ws.on('close', () => {
        clients.delete(ws);
        console.log('Client disconnected. Total clients:', clients.size);
    });
    
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        clients.delete(ws);
    });
});

// Broadcast message to all connected clients
function broadcast(message) {
    const data = JSON.stringify(message);
    clients.forEach(client => {
        if (client.readyState === client.OPEN) {
            try {
                client.send(data);
            } catch (error) {
                console.error('Error sending message to client:', error);
                clients.delete(client);
            }
        }
    });
}

// API Routes
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/api/generate', upload.single('image'), async (req, res) => {
    try {
        const {
            mode,
            prompt,
            video_length = '21',
            guidance_scale = '1.2',
            inference_steps = '4',
            quantization = 'false',
            low_memory = 'false',
            seed
        } = req.body;

        // Validate required fields
        if (!prompt || !prompt.trim()) {
            return res.status(400).json({
                success: false,
                error: 'Prompt is required'
            });
        }

        if (mode === 'image-to-video' && !req.file) {
            return res.status(400).json({
                success: false,
                error: 'Image is required for image-to-video mode'
            });
        }

        // Generate unique job ID
        const jobId = uuidv4();
        const outputDir = path.join(__dirname, '../outputs/web_interface', jobId);
        
        // Ensure output directory exists
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        // Prepare command arguments
        const args = [
            'test_t2v.py',
            '--config', 'configs/test/4_step_t2v.yaml',
            '--quantization', quantization,
            '--low_memory', low_memory
        ];

        // Create temporary config for this generation
        const configPath = path.join(outputDir, 'config.yaml');
        const config = {
            debug: true,
            seed: seed ? parseInt(seed) : Math.floor(Math.random() * 100000),
            exp_name: null,
            mode: 'predict',
            stage: 2,
            n_epochs: null,
            cache_dir: 'cache',
            ckpt_dir: 'outputs/ckpt',
            is_inference: true,
            resume_ckpt: '/home/efficiency/easytool/dmd_4_step_i2v.pth',
            resume_state_dict: true,
            test_data: {
                height: 540,
                width: 960,
                image_paths_and_scales: [[
                    req.file ? req.file.path : 'test_examples/top_1.png',
                    parseInt(video_length),
                    parseFloat(guidance_scale),
                    prompt.trim()
                ]]
            },
            inference: {
                inversion: false,
                output_dir: outputDir,
                num_inference_steps: parseInt(inference_steps),
                guidance_scale: parseFloat(guidance_scale),
                repeat_times: 1,
                quantization: quantization === 'true',
                low_memory: low_memory === 'true'
            }
        };

        // Write config file
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2));

        // Update args to use custom config
        args[2] = configPath;

        // Broadcast initial progress
        broadcast({
            type: 'progress',
            jobId,
            progress: 0,
            status: 'Initializing model...',
            logs: `Starting generation for: "${prompt}"`
        });

        // Start the Python process
        const pythonProcess = spawn('python', args, {
            cwd: path.join(__dirname, '../'),
            env: {
                ...process.env,
                CUDA_VISIBLE_DEVICES: '0',
                USE_FLASH_ATTENTION3: '0'
            }
        });

        let progress = 0;
        const progressSteps = [
            'Loading model weights...',
            'Processing input...',
            'Generating frames...',
            'Encoding video...',
            'Finalizing output...'
        ];
        let currentStep = 0;

        // Handle stdout
        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString();
            console.log('Python stdout:', output);
            
            // Update progress based on output
            if (output.includes('Loading')) {
                progress = Math.min(progress + 10, 90);
                currentStep = Math.min(currentStep + 1, progressSteps.length - 1);
            }
            
            broadcast({
                type: 'progress',
                jobId,
                progress,
                status: progressSteps[currentStep] || 'Processing...',
                logs: output.trim()
            });
        });

        // Handle stderr
        pythonProcess.stderr.on('data', (data) => {
            const error = data.toString();
            console.error('Python stderr:', error);
            
            // Don't broadcast all stderr as errors, some might be warnings
            if (error.toLowerCase().includes('error') || error.toLowerCase().includes('failed')) {
                broadcast({
                    type: 'error',
                    jobId,
                    message: error.trim()
                });
            } else {
                broadcast({
                    type: 'progress',
                    jobId,
                    progress,
                    status: progressSteps[currentStep] || 'Processing...',
                    logs: error.trim()
                });
            }
        });

        // Handle process completion
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                // Look for generated video file
                const videoFiles = fs.readdirSync(outputDir).filter(file => 
                    file.endsWith('.mp4') || file.endsWith('.avi') || file.endsWith('.mov')
                );
                
                if (videoFiles.length > 0) {
                    const videoFile = videoFiles[0];
                    const videoUrl = `/outputs/web_interface/${jobId}/${videoFile}`;
                    
                    broadcast({
                        type: 'complete',
                        jobId,
                        result: {
                            success: true,
                            video_url: videoUrl,
                            resolution: '960x540',
                            job_id: jobId
                        }
                    });
                    
                    res.json({
                        success: true,
                        video_url: videoUrl,
                        resolution: '960x540',
                        job_id: jobId
                    });
                } else {
                    const errorMsg = 'Video generation completed but no output file found';
                    broadcast({
                        type: 'error',
                        jobId,
                        message: errorMsg
                    });
                    
                    res.status(500).json({
                        success: false,
                        error: errorMsg
                    });
                }
            } else {
                const errorMsg = `Generation process failed with exit code ${code}`;
                broadcast({
                    type: 'error',
                    jobId,
                    message: errorMsg
                });
                
                res.status(500).json({
                    success: false,
                    error: errorMsg
                });
            }
        });

        // Handle process errors
        pythonProcess.on('error', (error) => {
            console.error('Process error:', error);
            const errorMsg = `Failed to start generation process: ${error.message}`;
            
            broadcast({
                type: 'error',
                jobId,
                message: errorMsg
            });
            
            res.status(500).json({
                success: false,
                error: errorMsg
            });
        });

    } catch (error) {
        console.error('Generation error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Serve generated videos
app.use('/outputs', express.static(path.join(__dirname, '../outputs')));

// Serve demo videos (if they exist)
app.use('/demo', express.static(path.join(__dirname, '../demo')));

// Serve uploaded images
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// Catch-all route to serve the main app
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../index.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Server error:', error);
    
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                success: false,
                error: 'File too large. Maximum size is 10MB.'
            });
        }
    }
    
    res.status(500).json({
        success: false,
        error: 'Internal server error'
    });
});

const PORT = process.env.PORT || 3000;

server.listen(PORT, () => {
    console.log(`Magic 1-For-1 Web Interface running on http://localhost:${PORT}`);
    console.log('WebSocket server is ready for real-time updates');
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('Received SIGTERM, shutting down gracefully');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('Received SIGINT, shutting down gracefully');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});