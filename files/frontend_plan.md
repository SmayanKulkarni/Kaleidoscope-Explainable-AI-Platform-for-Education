# LearnLens Frontend-Only Migration Plan

This plan documents the transition from the static HTML prototypes to a dynamic, React-based Vite application with entirely *mocked* backend services.

## 1. Project Initialization & Scaffolding
- Initialize the frontend using `vite` with the `react` template in `client/`.
- Configure Tailwind CSS (v3) in the React app, faithfully porting the Stitch design tokens.
- Install necessary frontend dependencies: `react-router-dom`, `lucide-react` (or Material Symbols).

## 2. Frontend Architecture & Flow (No Backend)
- **Global Auth State**: Implement an `AuthContext` to persist the mock "session" in `localStorage` across reloads.
- **Service Layer Mocking**: Implement a `xaiService.js` and `authService.js`.
  - `authService.js` will simulate signup/login, hardcoding authentication and returning a user object with a `role` (student or instructor).
  - `xaiService.js` will return hardcoded mock data matching the future FastAPI schemas (e.g., FastSHAP impact metrics, Anchor rules, counterfactuals).
- **Role-Based Routing (`react-router-dom`)**:
  - `Login/Signup`: A unified page where a role is selected during sign-up. 
  - `StudentDashboard`: Protected route, requires 'student' or 'instructor' role.
  - `InstructorDashboard`: Protected route, requires 'instructor' role.
  - `WhatIfExplorer`: Access strictly limited to instructors via a `<ProtectedRoute>`.
  - `ActionPlan`: Available to both, but the "Try in What-If" CTA is disabled for students.

## 3. UI/UX Component Migration
- Extract common `Navbar` and `Sidebar` layouts. Make the role display read-only mapping to the AuthContext.
- Convert the 5 static prototype screens into interactive JSX components.
- Add real React state interactivity to the What-If sliders to dynamically compute mock risk values.
