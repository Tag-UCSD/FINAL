import React from 'react';

export const Button = ({ children, onClick, variant = 'primary', disabled = false, className = '' }) => {
    const baseStyle = "px-4 py-2 rounded font-semibold transition-colors duration-200 disabled:opacity-50 flex items-center justify-center gap-2";
    const variants = {
        primary: "bg-blue-600 hover:bg-blue-700 text-white shadow-sm",
        secondary: "bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 shadow-sm",
        danger: "bg-red-600 hover:bg-red-700 text-white shadow-sm",
        ghost: "bg-transparent hover:bg-gray-100 text-gray-600"
    };

    return (
        <button 
            onClick={onClick} 
            disabled={disabled}
            className={`${baseStyle} ${variants[variant]} ${className}`}
        >
            {children}
        </button>
    );
};