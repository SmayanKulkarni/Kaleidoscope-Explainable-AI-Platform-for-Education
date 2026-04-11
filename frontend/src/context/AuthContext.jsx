import React, { createContext, useContext, useState, useEffect } from 'react';
import { login as apiLogin, register as apiRegister, getMe } from '../api/auth';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('ll_token');
    if (!token) { setLoading(false); return; }

    const withTimeout = (promise, ms = 8000) =>
      Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new Error('Auth bootstrap timeout')), ms)),
      ]);

    withTimeout(getMe())
      .then((me) => setUser(me))
      .catch(() => {
        localStorage.removeItem('ll_token');
        localStorage.removeItem('ll_user');
        localStorage.removeItem('ll_learner_id');
      })
      .finally(() => setLoading(false));
  }, []);

  const login = async (username, password) => {
    const tokenData = await apiLogin(username, password);
    localStorage.setItem('ll_token', tokenData.access_token);
    if (tokenData.learner_id) localStorage.setItem('ll_learner_id', tokenData.learner_id);
    const me = await getMe();
    localStorage.setItem('ll_user', JSON.stringify(me));
    setUser(me);
    return { ...me, role: tokenData.role };
  };

  const signup = async (body) => {
    await apiRegister(body);
    return login(body.username, body.password);
  };

  const logout = () => {
    localStorage.removeItem('ll_token');
    localStorage.removeItem('ll_user');
    localStorage.removeItem('ll_learner_id');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
