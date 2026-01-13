import React, { useEffect, useState } from 'react';
import { booksService, recommendationService } from '../services/api';
import type { Book, User } from '../services/api';
import BookCard from '../components/BookCard';
import { Loader2, ChevronLeft, ChevronRight } from 'lucide-react';

interface HomeProps {
  user: User | null;
}

const Home: React.FC<HomeProps> = ({ user }) => {
  const [books, setBooks] = useState<Book[]>([]);
  const [recommendations, setRecommendations] = useState<Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0); // Approximate

  useEffect(() => {
    fetchBooks();
  }, [page]);

  useEffect(() => {
    if (user) {
      fetchRecommendations();
    }
  }, [user]);

  const fetchBooks = async () => {
    try {
      setLoading(true);
      const data = await booksService.getBooks(page, 20); // 20 per page
      setBooks(data.books);
      // Backend returns total_count, let's assume limit 20
      setTotalPages(Math.ceil(data.total_count / 20));
    } catch (error) {
      console.error("Failed to fetch books", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendations = async () => {
    try {
      if (!user) return;
      const data = await recommendationService.getRecommendations(user.user_id.toString());
      setRecommendations(data);
    } catch (error) {
      console.error("Failed to fetch recommendations", error);
    }
  };

  return (
    <div className="bg-[#141414] min-h-screen text-white pb-20">
      {/* Hero Section Placeholder */}
      <div className="relative h-[50vh] w-full bg-gradient-to-r from-black to-gray-900 flex items-center justify-center mb-10">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1481627834876-b7833e8f5570?ixlib=rb-1.2.1&auto=format&fit=crop&w=2000&q=80')] bg-cover bg-center opacity-30"></div>
        <div className="relative z-10 text-center px-4">
          <h1 className="text-5xl md:text-7xl font-bold mb-4 drop-shadow-lg">Discover Your Next Story</h1>
          <p className="text-xl md:text-2xl text-gray-200">Personalized recommendations powered by AI</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">

        {/* Recommendations Section */}
        {user && recommendations.length > 0 && (
          <section>
             <h2 className="text-2xl font-semibold mb-4 text-white hover:text-red-600 transition-colors cursor-pointer">Top Picks for You</h2>
             <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {recommendations.map(book => (
                  <BookCard key={book.book_id} book={book} />
                ))}
             </div>
          </section>
        )}

        {/* All Books Section */}
        <section>
          <h2 className="text-2xl font-semibold mb-4 text-white">Trending Now</h2>
          {loading ? (
             <div className="flex justify-center py-20">
                <Loader2 className="h-10 w-10 animate-spin text-red-600" />
             </div>
          ) : (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {books.map(book => (
                  <BookCard key={book.book_id} book={book} />
                ))}
              </div>

              {/* Pagination */}
              <div className="flex justify-center items-center space-x-4 mt-10">
                <button
                  disabled={page === 1}
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  className="p-2 rounded-full bg-gray-800 hover:bg-gray-700 disabled:opacity-50 transition-colors"
                >
                  <ChevronLeft className="h-6 w-6" />
                </button>
                <span className="text-gray-400">Page {page} of {totalPages}</span>
                 <button
                  disabled={page >= totalPages}
                  onClick={() => setPage(p => p + 1)}
                  className="p-2 rounded-full bg-gray-800 hover:bg-gray-700 disabled:opacity-50 transition-colors"
                >
                  <ChevronRight className="h-6 w-6" />
                </button>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
};

export default Home;
