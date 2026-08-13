import React from 'react';
import { createRoot } from 'react-dom/client';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';
import App from './App';
import './styles/globals.css';
// Shared design tokens + theme factory (src/theme.ts): same palette values as
// before, plus semantic tokens, motion durations, skeleton defaults and
// keyboard-focus visibility.
import { buildTheme } from './theme';

const theme = buildTheme();

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
