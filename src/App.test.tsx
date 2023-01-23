import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn dagger link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn dagger/i);
  expect(linkElement).toBeInTheDocument();
});
