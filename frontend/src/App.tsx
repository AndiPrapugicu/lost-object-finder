import { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Chip,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Grid,
  Alert,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';
import VideoFeed from './components/VideoFeed';
import './App.css';

// Dark theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4caf50',
    },
    secondary: {
      main: '#2196f3',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

// Common objects that YOLOv8 can detect
const SUGGESTED_OBJECTS = [
  'phone',
  'laptop',
  'mouse',
  'keyboard',
  'bottle',
  'cup',
  'book',
  'scissors',
  'remote',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'clock',
  'vase',
];

function App() {
  const [targetObject, setTargetObject] = useState('');
  const [activeSearch, setActiveSearch] = useState('');
  const [inputValue, setInputValue] = useState('');

  const handleSearch = () => {
    if (inputValue.trim()) {
      setActiveSearch(inputValue.trim().toLowerCase());
      setTargetObject(inputValue.trim().toLowerCase());
    }
  };

  const handleClear = () => {
    setActiveSearch('');
    setTargetObject('');
    setInputValue('');
  };

  const handleSuggestedClick = (object: string) => {
    setInputValue(object);
    setActiveSearch(object);
    setTargetObject(object);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%)',
          py: 4,
        }}
      >
        <Container maxWidth="lg">
          {/* Header */}
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Typography
              variant="h2"
              sx={{
                fontWeight: 800,
                background: 'linear-gradient(45deg, #4caf50 30%, #2196f3 90%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 1,
              }}
            >
              🔍 Lost Object Finder
            </Typography>
            <Typography variant="h6" sx={{ color: '#999', mb: 3 }}>
              Real-time YOLOv8 Object Detection
            </Typography>

            {/* Search Input */}
            <Paper
              elevation={4}
              sx={{
                p: 3,
                backgroundColor: '#1a1a1a',
                border: '1px solid #333',
                maxWidth: '600px',
                margin: '0 auto',
              }}
            >
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  placeholder="What are you looking for? (e.g., phone, keys, bottle)"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      backgroundColor: '#0a0a0a',
                    },
                  }}
                />
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleSearch}
                  disabled={!inputValue.trim()}
                  startIcon={<SearchIcon />}
                  sx={{
                    minWidth: '120px',
                    fontWeight: 600,
                  }}
                >
                  Search
                </Button>
                {activeSearch && (
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={handleClear}
                    startIcon={<ClearIcon />}
                    color="error"
                  >
                    Clear
                  </Button>
                )}
              </Box>

              {/* Suggested Objects */}
              <Box>
                <Typography
                  variant="caption"
                  sx={{ color: '#999', display: 'block', mb: 1 }}
                >
                  Quick suggestions:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {SUGGESTED_OBJECTS.map((obj) => (
                    <Chip
                      key={obj}
                      label={obj}
                      onClick={() => handleSuggestedClick(obj)}
                      sx={{
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        '&:hover': {
                          backgroundColor: '#4caf50',
                          transform: 'scale(1.05)',
                        },
                        backgroundColor:
                          activeSearch === obj ? '#4caf50' : undefined,
                      }}
                    />
                  ))}
                </Box>
              </Box>
            </Paper>

            {/* Active Search Banner */}
            {activeSearch && (
              <Alert
                severity="info"
                sx={{
                  mt: 3,
                  maxWidth: '600px',
                  margin: '16px auto 0',
                  backgroundColor: 'rgba(33, 150, 243, 0.1)',
                  color: '#fff',
                }}
              >
                <Typography variant="body1">
                  🎯 Actively searching for: <strong>{activeSearch}</strong>
                </Typography>
                <Typography variant="caption">
                  Look at the camera - green boxes indicate your target!
                </Typography>
              </Alert>
            )}
          </Box>

          {/* Video Feed */}
          <VideoFeed
            targetObject={targetObject}
            backendUrl="http://localhost:8000"
          />

          {/* Info Section */}
          <Paper
            elevation={2}
            sx={{
              mt: 4,
              p: 3,
              backgroundColor: '#1a1a1a',
              border: '1px solid #333',
            }}
          >
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom sx={{ color: '#4caf50' }}>
                  🎥 How it works
                </Typography>
                <Typography variant="body2" sx={{ color: '#999' }}>
                  Our system uses YOLOv8 to detect objects in real-time. Simply
                  type what you're looking for, and the camera will highlight it
                  with a green box when found.
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom sx={{ color: '#2196f3' }}>
                  🟢 Green = Found
                </Typography>
                <Typography variant="body2" sx={{ color: '#999' }}>
                  When your target object is detected, it's highlighted with a
                  bright green box and "FOUND!" label. Other objects appear in
                  red.
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom sx={{ color: '#ff9800' }}>
                  ⚡ Performance
                </Typography>
                <Typography variant="body2" sx={{ color: '#999' }}>
                  Optimized to run at &gt;15 FPS with minimal latency. Check the
                  FPS counter in the bottom-left of the video feed.
                </Typography>
              </Grid>
            </Grid>
          </Paper>

          {/* Footer */}
          <Box sx={{ textAlign: 'center', mt: 4, pb: 2 }}>
            <Typography variant="caption" sx={{ color: '#666' }}>
              Built with FastAPI + YOLOv8 + React + TypeScript | Computer Vision
              Project 2025
            </Typography>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;

