import React from 'react';
import { LayoutGrid } from 'lucide-react';

export const Header = ({ title, appName }) => {
    return (
        <header className="bg-enterprise-blue text-white p-4 shadow-md flex justify-between items-center z-50">
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center font-bold shadow-inner">
                    <LayoutGrid size={18} />
                </div>
                <div>
                    <h1 className="text-lg font-bold leading-none tracking-tight">{appName}</h1>
                    <span className="text-xs text-gray-400 font-mono">v3.0 Enterprise</span>
                </div>
            </div>
            <div className="flex items-center gap-4">
                <div className="text-sm font-medium bg-surface-dark px-3 py-1 rounded border border-gray-700">
                    {title}
                </div>
                <div className="w-8 h-8 rounded-full bg-gray-600 border-2 border-gray-500"></div>
            </div>
        </header>
    );
};