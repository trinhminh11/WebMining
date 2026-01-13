import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, LogOut } from 'lucide-react';
import type { User } from '../services/api';

interface NavbarProps {
  user: User | null;
  onLogout: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ user, onLogout }) => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // Implement search logic or navigation
    console.log('Searching for:', searchTerm);
  };

  return (
    <nav className="fixed top-0 w-full z-50 bg-[#141414] bg-opacity-90 backdrop-blur-sm border-b border-gray-800 transition-all duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-red-600 text-2xl font-bold tracking-wider uppercase">
              BookFlix
            </Link>
            <div className="ml-10 flex items-baseline space-x-4">
              <Link to="/" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
                Home
              </Link>
              <Link to="/books" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
                All Books
              </Link>
            </div>
          </div>

          <div className="flex items-center space-x-6">
            <form onSubmit={handleSearch} className="relative group flex items-center">
              <div className="relative overflow-hidden rounded-full transition-all duration-300 w-0 group-hover:w-64 group-focus-within:w-64">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  className="w-full bg-black bg-opacity-50 text-gray-300 border border-gray-700 rounded-full pl-10 pr-4 py-1.5 focus:outline-none focus:border-gray-500 focus:text-white focus:bg-black"
                  placeholder="Titles, authors, genres"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
               <button type="button" className="ml-2 group-hover:hidden group-focus-within:hidden">
                 <Search className="h-6 w-6 text-white cursor-pointer" />
               </button>
            </form>

            <div className="flex items-center">
              {user ? (
                <div className="flex items-center space-x-4">
                  <span className="text-gray-300 text-sm">
                    {user.user_id}
                  </span>
                  <button
                    onClick={onLogout}
                    className="text-gray-300 hover:text-white transition-colors"
                  >
                    <LogOut className="h-5 w-5" />
                  </button>
                </div>
              ) : (
                <Link
                  to="/login"
                  className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded font-medium transition-colors"
                >
                  Sign In
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
