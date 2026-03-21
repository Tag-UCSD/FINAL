import React from 'react';

export const Toggle = ({ checked, onChange, label, danger = false }) => {
    return (
        <label className="flex items-center cursor-pointer">
            <div className="relative">
                <input 
                    type="checkbox" 
                    className="sr-only" 
                    checked={checked} 
                    onChange={(e) => onChange(e.target.checked)} 
                />
                <div className={`block w-12 h-7 rounded-full transition-colors ${
                    checked 
                        ? (danger ? 'bg-red-600' : 'bg-green-500') 
                        : 'bg-gray-300'
                }`}></div>
                <div className={`absolute left-1 top-1 bg-white w-5 h-5 rounded-full transition-transform shadow-sm ${
                    checked ? 'transform translate-x-5' : ''
                }`}></div>
            </div>
            {label && <span className="ml-3 text-sm font-medium text-gray-700">{label}</span>}
        </label>
    );
};