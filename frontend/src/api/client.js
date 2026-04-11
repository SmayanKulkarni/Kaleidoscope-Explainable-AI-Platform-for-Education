import axios from 'axios';

// In production the React app is served from the same origin as the API
// (FastAPI serves the built static files), so we use a relative base URL.
// For local dev via `npm run dev`, Vite's proxy rewrites /api → :8000, so
// we also don't need an absolute URL there.
// Override with VITE_API_URL only if the frontend is hosted separately.
const BASE_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? '/api' : '');


export const client = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30_000,
});

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('ll_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

client.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('ll_token');
      localStorage.removeItem('ll_user');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  },
);
