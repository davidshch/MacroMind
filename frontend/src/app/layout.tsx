import React from 'react';

const Layout: React.FC = ({ children }) => {
    return (
        <div>
            <header>
                <h1>MacroMind</h1>
                {/* Add navigation links here */}
            </header>
            <main>{children}</main>
            <footer>
                <p>&copy; {new Date().getFullYear()} MacroMind. All rights reserved.</p>
            </footer>
        </div>
    );
};

export default Layout;