import React from 'react';
import { Star } from 'lucide-react';
import type { Book } from '../services/api';
import { motion } from 'framer-motion';

interface BookCardProps {
  book: Book;
}

const BookCard: React.FC<BookCardProps> = ({ book }) => {
  return (
    <motion.div
      className="relative group bg-[#181818] rounded-md overflow-hidden cursor-pointer shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 hover:z-10"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="aspect-[2/3] relative">
        <img
          src={book.image_url_l || book.image_url_m}
          alt={book.title}
          className="w-full h-full object-cover"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
          <h3 className="text-white font-bold text-sm line-clamp-2">{book.title}</h3>
          <p className="text-gray-400 text-xs mt-1">{book.author}</p>
          <div className="flex items-center mt-2 space-x-1">
             <Star className="h-3 w-3 text-yellow-500 fill-current" />
             <span className="text-green-400 text-xs font-bold">{book.average_rating?.toFixed(1) || 'N/A'}</span>
             <span className="text-gray-500 text-xs">({book.rating_count || 0})</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default BookCard;
