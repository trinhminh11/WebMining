import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface User {
  user_id: number;
  location?: string;
  age?: number;
}

export interface Book {
  book_id: string;
  title: string;
  author: string;
  year_of_publication: number;
  publisher: string;
  image_url_s: string;
  image_url_m: string;
  image_url_l: string;
  average_rating?: number;
  rating_count?: number;
}

export interface BookListResponse {
  books: Book[];
  total_count: number;
  page: number;
  limit: number;
}

export const authService = {
  login: async (userId: string, password: string): Promise<User> => {
    // Backend expects user_id as int if the schema uses int.
    // Wait, the SQL schema says BIGINT, but the schema in `check_stats.py` or similar might convert.
    // Let's assume the API expects the raw value.
    // Note: The backend `login` takes `UserLogin` schema.
    const response = await api.post<User>('/auth/login', { user_id: parseInt(userId), password });
    return response.data;
  },
  register: async (userId: string, password: string, location?: string, age?: number): Promise<User> => {
    const response = await api.post<User>('/auth/register', {
        user_id: parseInt(userId),
        password,
        location: location || "Unknown",
        age: age || 18
    });
    return response.data;
  },
};

export const booksService = {
  getBooks: async (page = 1, limit = 20): Promise<BookListResponse> => {
    const response = await api.get<BookListResponse>('/books', { params: { page, limit } });
    return response.data;
  },
  getBook: async (bookId: string): Promise<Book> => {
    const response = await api.get<Book>(`/books/${bookId}`);
    return response.data;
  },
};

export const ratingService = {
  rateBook: async (userId: string, bookId: string, rating: number) => {
    const response = await api.post('/ratings', { user_id: parseInt(userId), book_id: bookId, rating });
    return response.data;
  },
};

export const recommendationService = {
  getRecommendations: async (userId: string): Promise<Book[]> => {
    const response = await api.get<Book[]>(`/recommendations/${userId}`);
    return response.data;
  },
};

export default api;
