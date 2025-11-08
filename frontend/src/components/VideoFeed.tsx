import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Fade,
  Chip,
  Button,
} from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import SearchIcon from '@mui/icons-material/Search';

// Pulse animation for "FOUND!" banner
const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.9;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

// Styled components with dark mode theme
const VideoContainer = styled(Paper)(() => ({
  position: 'relative',
  width: '100%',
  maxWidth: '800px',
  margin: '0 auto',
  backgroundColor: '#1a1a1a',
  borderRadius: '16px',
  overflow: 'hidden',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
  border: '2px solid #333',
}));

const StyledVideo = styled('img')({
  width: '100%',
  height: 'auto',
  display: 'block',
  objectFit: 'contain',
  minHeight: '400px',
  backgroundColor: '#000',
});

const OverlayBanner = styled(Box)<{ found?: boolean }>(({ found }) => ({
  position: 'absolute',
  top: '20px',
  left: '50%',
  transform: 'translateX(-50%)',
  padding: '12px 24px',
  borderRadius: '12px',
  backgroundColor: found ? 'rgba(76, 175, 80, 0.95)' : 'rgba(33, 33, 33, 0.95)',
  backdropFilter: 'blur(10px)',
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
  animation: found ? `${pulse} 1.5s ease-in-out infinite` : 'none',
  border: found ? '2px solid #4caf50' : '2px solid #444',
  zIndex: 10,
}));

const StatusChip = styled(Chip)(() => ({
  position: 'absolute',
  bottom: '20px',
  left: '20px',
  backgroundColor: 'rgba(33, 33, 33, 0.9)',
  color: '#fff',
  backdropFilter: 'blur(10px)',
  fontWeight: 600,
  fontSize: '0.9rem',
}));

const LoadingContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: '400px',
  gap: '20px',
  backgroundColor: '#1a1a1a',
});

interface VideoFeedProps {
  targetObject: string;
  backendUrl?: string;
}

/**
 * VideoFeed Component
 * 
 * Displays real-time MJPEG video stream from FastAPI backend with YOLOv8 detection.
 * 
 * Props:
 *   - targetObject: Name of object to find (e.g., "phone", "keys", "bottle")
 *   - backendUrl: Backend API URL (default: http://localhost:8000)
 * 
 * Features:
 *   - Real-time video streaming with MJPEG
 *   - Animated "FOUND!" overlay when target object is detected
 *   - "Searching..." overlay when object is not found
 *   - Error handling for connection failures
 *   - Dark mode Material UI styling
 *   - Responsive design
 */
const VideoFeed: React.FC<VideoFeedProps> = ({
  targetObject,
  backendUrl = 'http://localhost:8000',
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [cameraPermissionGranted, setCameraPermissionGranted] = useState(false);
  const [isCheckingCamera, setIsCheckingCamera] = useState(false);
  const [streamEnabled, setStreamEnabled] = useState(false);

  // Construct stream URL with target object parameter
  const streamUrl = targetObject
    ? `${backendUrl}/track?target_object=${encodeURIComponent(targetObject)}`
    : `${backendUrl}/track`;

  // Add timestamp to force refresh when re-enabling camera
  const streamUrlWithTimestamp = streamEnabled ? `${streamUrl}&t=${Date.now()}` : '';

  useEffect(() => {
    // Reset states when target object changes
    setIsLoading(true);
    setHasError(false);
    setErrorMessage('');
    setIsStreamActive(false);
  }, [targetObject]);

  // Check camera availability on mount
  const checkCameraAvailability = async () => {
    setIsCheckingCamera(true);
    try {
      const response = await fetch(`${backendUrl}/list_cameras`);
      const data = await response.json();
      
      if (data.available_cameras && data.available_cameras.length > 0) {
        setCameraPermissionGranted(true);
        setStreamEnabled(true); // Enable stream
        setIsLoading(true); // Start loading the stream
      } else {
        setHasError(true);
        setErrorMessage('No camera detected. Please connect a camera and try again.');
      }
    } catch {
      setHasError(true);
      setErrorMessage('Cannot connect to backend. Make sure the server is running.');
    } finally {
      setIsCheckingCamera(false);
    }
  };

  const stopCamera = () => {
    setStreamEnabled(false);
    setIsStreamActive(false);
    setIsLoading(false);
    console.log('📹 Camera stopped by user');
  };

  const restartCamera = () => {
    setStreamEnabled(true);
    setIsLoading(true);
    setHasError(false);
    console.log('📹 Camera restarted by user');
  };

  const handleImageLoad = () => {
    setIsLoading(false);
    setHasError(false);
    setIsStreamActive(true);
    console.log('✅ Video stream connected successfully');
  };

  const handleImageError = () => {
    setIsLoading(false);
    setHasError(true);
    setIsStreamActive(false);
    setErrorMessage(
      'Failed to connect to video stream. Make sure the backend is running at ' + backendUrl
    );
    console.error('❌ Failed to load video stream from:', streamUrl);
  };

  return (
    <Box sx={{ width: '100%', py: 2 }}>
      <VideoContainer elevation={8}>
        {/* Camera Permission Request */}
        {!cameraPermissionGranted && !hasError && (
          <LoadingContainer>
            <VideocamIcon sx={{ fontSize: 80, color: '#4caf50', mb: 2 }} />
            <Typography variant="h5" sx={{ color: '#fff', mb: 2 }}>
              Camera Access Required
            </Typography>
            <Typography variant="body1" sx={{ color: '#999', mb: 3, textAlign: 'center', px: 3 }}>
              This application needs access to your camera to detect objects in real-time.
            </Typography>
            <Button
              variant="contained"
              size="large"
              startIcon={<VideocamIcon />}
              onClick={checkCameraAvailability}
              disabled={isCheckingCamera}
              sx={{
                backgroundColor: '#4caf50',
                '&:hover': { backgroundColor: '#45a049' },
                px: 4,
                py: 1.5,
                fontSize: '1.1rem',
                fontWeight: 600,
              }}
            >
              {isCheckingCamera ? (
                <>
                  <CircularProgress size={20} sx={{ mr: 1, color: '#fff' }} />
                  Checking Camera...
                </>
              ) : (
                'Enable Camera'
              )}
            </Button>
            <Typography variant="caption" sx={{ color: '#666', mt: 2, textAlign: 'center' }}>
              The backend server will access your camera through OpenCV
            </Typography>
          </LoadingContainer>
        )}

        {/* Loading State */}
        {cameraPermissionGranted && isLoading && streamEnabled && (
          <LoadingContainer>
            <CircularProgress size={60} sx={{ color: '#4caf50' }} />
            <Typography variant="h6" sx={{ color: '#fff' }}>
              Connecting to camera...
            </Typography>
            <Typography variant="body2" sx={{ color: '#999' }}>
              {targetObject ? `Looking for: ${targetObject}` : 'No target set'}
            </Typography>
          </LoadingContainer>
        )}

        {/* Camera Stopped State */}
        {cameraPermissionGranted && !streamEnabled && (
          <LoadingContainer>
            <VideocamOffIcon sx={{ fontSize: 80, color: '#666', mb: 2 }} />
            <Typography variant="h5" sx={{ color: '#fff', mb: 2 }}>
              Camera Stopped
            </Typography>
            <Typography variant="body2" sx={{ color: '#999', mb: 3 }}>
              Click below to restart the camera feed
            </Typography>
            <Button
              variant="contained"
              size="large"
              startIcon={<VideocamIcon />}
              onClick={restartCamera}
              sx={{
                backgroundColor: '#4caf50',
                '&:hover': { backgroundColor: '#45a049' },
                px: 4,
                py: 1.5,
                fontSize: '1rem',
                fontWeight: 600,
              }}
            >
              Restart Camera
            </Button>
          </LoadingContainer>
        )}

        {/* Error State */}
        {hasError && (
          <Box sx={{ p: 3 }}>
            <Alert
              severity="error"
              sx={{
                backgroundColor: 'rgba(211, 47, 47, 0.1)',
                color: '#fff',
                '& .MuiAlert-icon': { color: '#f44336' },
              }}
            >
              <Typography variant="body1" gutterBottom>
                <strong>Camera Connection Error</strong>
              </Typography>
              <Typography variant="body2">{errorMessage}</Typography>
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                Make sure:
                <br />
                1. Backend server is running (http://localhost:8000)
                <br />
                2. Camera permissions are granted
                <br />
                3. No other app is using the camera
              </Typography>
            </Alert>
          </Box>
        )}

        {/* Video Stream */}
        {cameraPermissionGranted && streamEnabled && (
          <Box sx={{ position: 'relative', display: isLoading ? 'none' : 'block' }}>
            <StyledVideo
              src={streamUrlWithTimestamp}
              alt="Live video feed"
              onLoad={handleImageLoad}
              onError={handleImageError}
            />

          {/* Overlay: Searching or No Target */}
          {isStreamActive && targetObject && (
            <Fade in timeout={500}>
              <OverlayBanner>
                <SearchIcon sx={{ fontSize: 28, color: '#fff' }} />
                <Typography
                  variant="h6"
                  sx={{
                    color: '#fff',
                    fontWeight: 700,
                    textShadow: '0 2px 4px rgba(0,0,0,0.3)',
                  }}
                >
                  Searching for: {targetObject}
                </Typography>
              </OverlayBanner>
            </Fade>
          )}

          {/* No Target Set */}
          {isStreamActive && !targetObject && (
            <Fade in timeout={500}>
              <OverlayBanner>
                <VideocamIcon sx={{ fontSize: 28, color: '#fff' }} />
                <Typography
                  variant="h6"
                  sx={{
                    color: '#fff',
                    fontWeight: 700,
                  }}
                >
                  Live Camera Feed
                </Typography>
              </OverlayBanner>
            </Fade>
          )}

          {/* Status Chip */}
          {isStreamActive && (
            <StatusChip
              icon={<VideocamIcon sx={{ color: '#4caf50' }} />}
              label="LIVE"
            />
          )}

          {/* Stop Camera Button */}
          {isStreamActive && (
            <Box
              sx={{
                position: 'absolute',
                bottom: '20px',
                right: '20px',
              }}
            >
              <Button
                variant="contained"
                size="small"
                onClick={stopCamera}
                startIcon={<VideocamOffIcon />}
                sx={{
                  backgroundColor: 'rgba(211, 47, 47, 0.9)',
                  backdropFilter: 'blur(10px)',
                  '&:hover': {
                    backgroundColor: 'rgba(183, 28, 28, 1)',
                  },
                  fontWeight: 600,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                }}
              >
                Stop Camera
              </Button>
            </Box>
          )}
          </Box>
        )}
      </VideoContainer>

      {/* Info Text */}
      {isStreamActive && (
        <Box sx={{ textAlign: 'center', mt: 2 }}>
          <Typography variant="body2" sx={{ color: '#999' }}>
            {targetObject
              ? 'Green box = Target found | Red boxes = Other objects'
              : 'Set a target object to start searching'}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default VideoFeed;
