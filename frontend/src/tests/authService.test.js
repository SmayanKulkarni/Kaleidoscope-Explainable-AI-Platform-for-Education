import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../services/api', () => ({
  api: { post: vi.fn(), get: vi.fn() },
}));

import { api } from '../services/api';
import { login, register, getMe, getStudents } from '../services/authService';

beforeEach(() => { vi.clearAllMocks(); });

describe('login', () => {
  it('calls POST /auth/login with { username, password }', async () => {
    api.post.mockResolvedValueOnce({ data: { access_token: 'tok', user_id: 'u1', role: 'student' } });
    await login('alice', 'secret');
    expect(api.post).toHaveBeenCalledWith('/auth/login', { username: 'alice', password: 'secret' });
  });

  it('returns the token response object', async () => {
    api.post.mockResolvedValueOnce({ data: { access_token: 'tok123', user_id: 'u2', role: 'instructor' } });
    const result = await login('bob', 'pass');
    expect(result.access_token).toBe('tok123');
    expect(result.role).toBe('instructor');
  });

  it('propagates API errors without swallowing them', async () => {
    api.post.mockRejectedValueOnce({ response: { status: 401, data: { detail: 'Invalid credentials' } } });
    await expect(login('bad', 'creds')).rejects.toMatchObject({ response: { status: 401 } });
  });
});

describe('register', () => {
  it('calls POST /auth/register with full payload', async () => {
    api.post.mockResolvedValueOnce({ data: { user_id: 'u3' } });
    const payload = { username: 'new_user', password: 'pw', role: 'student', full_name: 'New User' };
    await register(payload);
    expect(api.post).toHaveBeenCalledWith('/auth/register', payload);
  });
});

describe('getMe', () => {
  it('calls GET /auth/me', async () => {
    api.get.mockResolvedValueOnce({ data: { username: 'me', email: 'me@x.com', role: 'student' } });
    const result = await getMe();
    expect(api.get).toHaveBeenCalledWith('/auth/me');
    expect(result.username).toBe('me');
  });
});

describe('getStudents', () => {
  it('calls GET /auth/students with empty params when no courseId', async () => {
    api.get.mockResolvedValueOnce({ data: [] });
    await getStudents();
    expect(api.get).toHaveBeenCalledWith('/auth/students', { params: {} });
  });

  it('calls GET /auth/students with course_id param when courseId provided', async () => {
    api.get.mockResolvedValueOnce({ data: [] });
    await getStudents('COURSE_XYZ');
    expect(api.get).toHaveBeenCalledWith('/auth/students', { params: { course_id: 'COURSE_XYZ' } });
  });
});
