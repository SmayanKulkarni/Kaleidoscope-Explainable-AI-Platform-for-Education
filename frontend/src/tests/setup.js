import { vi } from 'vitest';

Object.defineProperty(window, 'location', {
  value: { href: '', pathname: '/test' },
  writable: true,
});
