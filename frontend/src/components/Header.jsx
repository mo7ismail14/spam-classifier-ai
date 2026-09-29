import { AppBar, Toolbar, Typography, Tabs, Tab, Box } from '@mui/material';
import { Security } from '@mui/icons-material';
import { useLocation, useNavigate } from 'react-router-dom';
import routes from '../routes';

const navRoutes = routes.filter((route) => route.showInNav);

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();

  const currentPath = navRoutes.some((route) => route.path === location.pathname)
    ? location.pathname
    : false;

  return (
    <AppBar position="static" elevation={2} sx={{ bgcolor: '#1a237e' }}>
      <Toolbar sx={{ gap: 2 }}>
        <Security />
        <Typography variant="h6" component="div" sx={{ fontWeight: 'bold', mr: 2 }}>
          SpamShield AI
        </Typography>

        <Box sx={{ flexGrow: 1 }} />

        <Tabs
          value={currentPath}
          onChange={(event, newPath) => navigate(newPath)}
          textColor="inherit"
          indicatorColor="secondary"
        >
          {navRoutes.map((route) => (
            <Tab key={route.path} value={route.path} label={route.label} />
          ))}
        </Tabs>
      </Toolbar>
    </AppBar>
  );
}
