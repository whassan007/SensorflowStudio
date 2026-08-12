import React from 'react';
import { createRoot } from 'react-dom/client';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import App from './App';
import './styles/globals.css';

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: { default: '#101418', paper: '#161b21' },
    primary: { main: '#4fc3f7' },
  },
  typography: {
    fontFamily: "'Inter', 'Helvetica Neue', Arial, sans-serif",
  },
});

const container = document.getElementById('root');
if (!container) throw new Error('Root element missing');
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
