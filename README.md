# Magic 1-For-1 Web Interface

A modern, responsive web interface for the Magic 1-For-1 video generation model. This interface provides an intuitive way to generate high-quality videos from text prompts or images.

## Features

### 🎬 Dual Generation Modes
- **Text-to-Video**: Generate videos directly from text descriptions
- **Image-to-Video**: Animate static images with text guidance

### ⚡ Advanced Controls
- **Video Length**: Choose from 21, 41, or 61 frames
- **Guidance Scale**: Fine-tune adherence to prompts (1.0-3.0)
- **Inference Steps**: Balance quality vs speed (4-30 steps)
- **Quantization**: Enable for faster generation on limited hardware
- **Low Memory Mode**: Optimize for systems with limited VRAM
- **Custom Seeds**: Reproduce specific results

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Real-time Progress**: Live updates during generation with WebSocket
- **Interactive Gallery**: Browse and replay previous generations
- **Drag & Drop**: Easy image uploading with preview
- **Dark/Light Themes**: Automatic theme adaptation

### 🚀 Performance Features
- **WebSocket Integration**: Real-time progress updates and logs
- **Efficient File Handling**: Optimized image upload and processing
- **Memory Management**: Smart resource allocation for different hardware
- **Error Recovery**: Graceful error handling with detailed feedback

## Quick Start

### Prerequisites
- Node.js 18+ installed
- Python 3.9+ with Magic 1-For-1 dependencies
- CUDA-compatible GPU (recommended)

### Installation

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Start the backend server:**
   ```bash
   npm run server
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

### Production Deployment

1. **Build the frontend:**
   ```bash
   npm run build
   ```

2. **Start the production server:**
   ```bash
   npm start
   ```

## Usage Guide

### Text-to-Video Generation

1. **Select Mode**: Choose "Text to Video" from the mode selector
2. **Enter Prompt**: Describe your desired video in detail
   - Example: "A cat playing piano in a cozy living room"
3. **Adjust Settings**: Configure advanced options if needed
4. **Generate**: Click "Generate Video" and wait for completion
5. **Download**: Save your generated video or add it to the gallery

### Image-to-Video Generation

1. **Select Mode**: Choose "Image to Video" from the mode selector
2. **Upload Image**: Drag and drop or click to upload a reference image
3. **Enter Prompt**: Describe how you want the image to move
   - Example: "The cat starts playing the piano keys"
4. **Configure**: Adjust generation parameters as needed
5. **Generate**: Start the generation process
6. **Review**: Preview and download your animated video

### Advanced Settings

- **Video Length**: Longer videos take more time but provide more content
- **Guidance Scale**: Higher values follow prompts more closely
- **Inference Steps**: More steps generally improve quality
- **Quantization**: Reduces memory usage and speeds up generation
- **Low Memory Mode**: Essential for GPUs with limited VRAM
- **Seed**: Use the same seed to reproduce identical results

## API Reference

### POST /api/generate

Generate a video from text prompt and optional image.

**Parameters:**
- `mode`: "text-to-video" or "image-to-video"
- `prompt`: Text description of desired video
- `image`: Image file (required for image-to-video mode)
- `video_length`: Number of frames (21, 41, or 61)
- `guidance_scale`: Guidance strength (1.0-3.0)
- `inference_steps`: Number of denoising steps (4-30)
- `quantization`: Enable quantization (boolean)
- `low_memory`: Enable low memory mode (boolean)
- `seed`: Random seed for reproducibility (optional)

**Response:**
```json
{
  "success": true,
  "video_url": "/outputs/web_interface/uuid/video.mp4",
  "resolution": "960x540",
  "job_id": "uuid"
}
```

### WebSocket /ws

Real-time updates during generation process.

**Message Types:**
- `progress`: Generation progress updates
- `complete`: Generation completed successfully
- `error`: Error occurred during generation

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
PORT=3000
NODE_ENV=development
PYTHON_PATH=/path/to/python
MODEL_PATH=/path/to/model/weights
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

The interface automatically configures the Magic 1-For-1 model based on your settings. Key configuration files:

- `configs/test/4_step_t2v.yaml`: Base configuration for 4-step generation
- Custom configs are generated per request for optimal performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Enable "Low Memory Mode"
   - Reduce video length
   - Enable quantization

2. **Slow Generation**
   - Enable quantization
   - Reduce inference steps
   - Use shorter video lengths

3. **WebSocket Connection Failed**
   - Check firewall settings
   - Ensure port 3000 is available
   - Restart the server

4. **Model Loading Errors**
   - Verify model weights are downloaded
   - Check CUDA installation
   - Ensure sufficient disk space

### Performance Optimization

- **GPU Memory**: Use quantization and low memory mode for GPUs with <8GB VRAM
- **Generation Speed**: 4-step inference provides good quality/speed balance
- **Quality**: Use 16-30 steps for highest quality output
- **Batch Processing**: Generate multiple videos sequentially rather than simultaneously

## Development

### Project Structure

```
├── index.html          # Main application HTML
├── style.css           # Comprehensive styling
├── script.js           # Frontend JavaScript logic
├── server/
│   └── index.js        # Express server with WebSocket
├── configs/            # Model configuration files
├── uploads/            # Temporary image uploads
└── outputs/            # Generated video outputs
```

### Adding Features

1. **Frontend**: Modify `script.js` and `style.css`
2. **Backend**: Update `server/index.js`
3. **Model Integration**: Adjust configuration in `configs/`

### Testing

```bash
# Test the API endpoint
curl -X POST http://localhost:3000/api/health

# Test WebSocket connection
wscat -c ws://localhost:3000/ws
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Magic 1-For-1 research initiative. Please refer to the main project license for usage terms.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the Magic 1-For-1 project documentation
- Open an issue on the project repository

---

**Magic 1-For-1**: Generating One Minute Video Clips within One Minute