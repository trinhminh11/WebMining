import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import type { User } from './services/api';

function App() {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // Check local storage for user
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (e) {
        console.error("Invalid user in localStorage");
        localStorage.removeItem('user');
      }
    }
  }, []);

  const handleLogin = (newUser: User) => {
    setUser(newUser);
    localStorage.setItem('user', JSON.stringify(newUser));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };

  return (
    <Router>
      <div className="bg-[#141414] min-h-screen text-white font-sans antialiased">
        <Navbar user={user} onLogout={handleLogout} />
        <div className="pt-16">
          <Routes>
            <Route path="/" element={<Home user={user} />} />
            <Route path="/login" element={!user ? <Login onLogin={handleLogin} /> : <Navigate to="/" />} />
            <Route path="/register" element={!user ? <Register onLogin={handleLogin} /> : <Navigate to="/" />} />
            {/* Fallback route */}
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
