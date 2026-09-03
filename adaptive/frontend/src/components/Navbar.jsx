import React from 'react';
import { NavLink } from 'react-router-dom';

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="brand">
        ⚡ AI Resume Intelligence Engine
      </div>
      <nav className="nav-links">
        <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>
          Upload
        </NavLink>
        <NavLink to="/extracted" className={({ isActive }) => (isActive ? 'active' : '')}>
          Intelligence Summary
        </NavLink>
        <NavLink to="/skills" className={({ isActive }) => (isActive ? 'active' : '')}>
          Technical Skills
        </NavLink>
        <NavLink to="/interview-dashboard" className={({ isActive }) => (isActive ? 'active' : '')}>
          Interview
        </NavLink>
      </nav>
    </header>
  );
}

