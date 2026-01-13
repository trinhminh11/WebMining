import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { authService } from '../services/api';
import type { User } from '../services/api';

interface LoginProps {
  onLogin: (user: User) => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const user = await authService.login(userId, password);
      onLogin(user);
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed');
    }
  };

  return (
    <div className="min-h-screen bg-black bg-opacity-90 flex items-center justify-center px-4 bg-[url('https://assets.nflxext.com/ffe/siteui/vlv3/f841d4c7-10e1-40af-bcae-07a3f8dc141a/f6d7434e-d6de-4185-a6d4-c77a2d08737b/US-en-20220502-popsignuptwoweeks-perspective_alpha_website_medium.jpg')] bg-cover bg-blend-overlay">
      <div className="max-w-md w-full bg-black bg-opacity-75 p-8 rounded-lg shadow-xl">
        <h2 className="text-3xl font-bold text-white mb-8">Sign In</h2>
        {error && (
          <div className="bg-[#e87c03] p-3 rounded mb-4 text-white text-sm">
            {error}
          </div>
        )}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="userId" className="sr-only">User ID</label>
            <input
              id="userId"
              type="text"
              required
              className="w-full bg-[#333] rounded px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:bg-[#454545]"
              placeholder="User ID (e.g., 276725)"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="password" className="sr-only">Password</label>
            <input
              id="password"
              type="password"
              required
              className="w-full bg-[#333] rounded px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:bg-[#454545]"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <button
            type="submit"
            className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 rounded transition duration-200"
          >
            Sign In
          </button>
        </form>
        <div className="mt-8 text-gray-400 text-sm">
          New to BookFlix? <Link to="/register" className="text-white hover:underline">Sign up now</Link>.
        </div>
      </div>
    </div>
  );
};

export default Login;
