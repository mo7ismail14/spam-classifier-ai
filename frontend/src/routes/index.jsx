import { lazy } from 'react';

// Central route registry. Add a new page by adding one entry here —
// the router (App.jsx) and the header nav (components/Header.jsx) both
// derive themselves from this list, so nothing else needs to change.
const routes = [
  {
    path: '/',
    label: 'Home',
    showInNav: true,
    component: lazy(() => import('../pages/Landing'))
  },
  {
    path: '/spam',
    label: 'Spam Classifier',
    showInNav: true,
    component: lazy(() => import('../pages/Spam'))
  },
  {
    path: '/loan',
    label: 'Loan Approval',
    showInNav: true,
    component: lazy(() => import('../pages/LoanApproval'))
  }
];

export default routes;
