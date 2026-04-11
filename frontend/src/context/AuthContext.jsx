import React, { createContext, useContext, useState, useEffect } from 'react';
import { loginMock, signupMock } from '../services/authService';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Restore mock session
    const stored = localStorage.getItem('ll_mock_user');
    if (stored) {
      setUser(JSON.parse(stored));
    }
    setLoading(false);
  }, []);

  const login = async (email, password) => {
    const u = await loginMock(email, password);
    localStorage.setItem('ll_mock_user', JSON.stringify(u));
    setUser(u);
    return u;
  };

  const signup = async (name, email, password, role) => {
    const u = await signupMock(name, email, password, role);
    localStorage.setItem('ll_mock_user', JSON.stringify(u));
    setUser(u);
    return u;
  };

  const logout = () => {
    localStorage.removeItem('ll_mock_user');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
