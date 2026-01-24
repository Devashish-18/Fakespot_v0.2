import React from 'react';
import { Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

export default function Navbar() {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <nav className="sticky top-0 z-50 glass-effect border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-10 h-10 gradient-primary rounded-lg flex items-center justify-center text-white font-bold">
              FS
            </div>
            <span className="text-xl font-bold text-dark hidden sm:inline">FAKESPOT</span>
          </Link>

          {/* Desktop Menu */}
          <div className="hidden md:flex gap-8">
            <Link to="/" className="text-dark hover:text-primary transition">Home</Link>
            <a href="/#features" className="text-dark hover:text-primary transition">Features</a>
            <a href="/#how-it-works" className="text-dark hover:text-primary transition">How It Works</a>
            <Link to="/analyzer" className="text-dark hover:text-primary transition">Analyze</Link>
          </div>

          <Link to="/analyzer" className="hidden md:block px-6 py-2 gradient-primary text-white rounded-lg hover:shadow-lg transition">
            Analyze Now
          </Link>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden pb-4 border-t border-gray-200">
            <Link to="/" className="block py-2 text-dark hover:text-primary transition">Home</Link>
            <a href="/#features" className="block py-2 text-dark hover:text-primary transition">Features</a>
            <a href="/#how-it-works" className="block py-2 text-dark hover:text-primary transition">How It Works</a>
            <Link to="/analyzer" className="block py-2 text-dark hover:text-primary transition">Analyze</Link>
          </div>
        )}
      </div>
    </nav>
  );
}
