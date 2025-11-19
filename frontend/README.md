# Model Leaderboard Frontend

This is a Next.js application built with shadcn/ui components for the Model Leaderboard.

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Building for Production

### For GitHub Pages

```bash
npm run build:docs
```

This will create static files in the `out/` directory and copy them to the `../docs/` directory for GitHub Pages deployment. It also creates a `.nojekyll` file to ensure proper deployment.

### For Regular Production Build

```bash
npm run build
```

This creates static files in the `out/` directory that can be used for other deployment methods.

## Project Structure

- `src/app/` - Next.js app router pages
- `src/components/ui/` - shadcn/ui components
- `src/lib/` - Utility functions
- `public/` - Static assets
