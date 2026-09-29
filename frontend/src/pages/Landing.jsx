import { Box, Container, Typography, Button, Grid, Card, CardContent, Chip, Stack } from '@mui/material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  RocketLaunch,
  Psychology,
  Speed,
  QueryStats,
  Security,
  Bolt
} from '@mui/icons-material';

const MotionBox = motion(Box);
const MotionTypography = motion(Typography);
const MotionButton = motion(Button);
const MotionCard = motion(Card);

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay, ease: 'easeOut' }
  })
};

const features = [
  {
    icon: <Psychology fontSize="large" />,
    title: 'Multiple ML Models',
    description: 'Compare Support Vector Machines and Logistic Regression side by side, both tuned with GridSearchCV and K-Fold cross-validation.'
  },
  {
    icon: <Bolt fontSize="large" />,
    title: 'Real-Time Classification',
    description: 'Paste any email or pick a held-out sample, choose a model, and get an instant spam/ham verdict with a confidence score.'
  },
  {
    icon: <QueryStats fontSize="large" />,
    title: 'Transparent Performance',
    description: 'Every prediction ships with the full confusion matrix — accuracy, precision, recall and F1 — computed on truly unseen test data.'
  }
];

const techStack = ['React', 'Vite', 'Material UI', 'Flask', 'scikit-learn', 'TF-IDF', 'SVM', 'Logistic Regression'];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <Box sx={{ overflowX: 'hidden' }}>
      {/* Hero Section */}
      <MotionBox
        sx={{
          position: 'relative',
          minHeight: { xs: '80vh', md: '90vh' },
          display: 'flex',
          alignItems: 'center',
          background: 'linear-gradient(135deg, #1a237e 0%, #3949ab 45%, #7c4dff 100%)',
          backgroundSize: '200% 200%',
          color: '#fff',
          overflow: 'hidden'
        }}
        animate={{ backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'] }}
        transition={{ duration: 16, repeat: Infinity, ease: 'linear' }}
      >
        {/* Floating decorative blobs */}
        <MotionBox
          sx={{
            position: 'absolute',
            width: 320,
            height: 320,
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.08)',
            top: -80,
            right: -60
          }}
          animate={{ y: [0, 30, 0], x: [0, -20, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
        />
        <MotionBox
          sx={{
            position: 'absolute',
            width: 220,
            height: 220,
            borderRadius: '50%',
            background: 'rgba(124,77,255,0.25)',
            bottom: -60,
            left: -40
          }}
          animate={{ y: [0, -25, 0], x: [0, 15, 0] }}
          transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
        />

        <Container maxWidth="md" sx={{ position: 'relative', textAlign: 'center', py: 8 }}>
          <MotionTypography
            variant="h2"
            sx={{ mb: 2, fontSize: { xs: '2.2rem', sm: '3rem', md: '3.5rem' } }}
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={0}
          >
            Spam Detection, Powered by AI
          </MotionTypography>

          <MotionTypography
            variant="h6"
            sx={{ mb: 5, fontWeight: 400, opacity: 0.9, maxWidth: 640, mx: 'auto' }}
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={0.15}
          >
            An end-to-end machine learning project that classifies emails as spam or ham,
            trained and benchmarked across multiple algorithms with full transparency into
            how each decision is made.
          </MotionTypography>

          <MotionBox
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={0.3}
          >
            <MotionButton
              variant="contained"
              size="large"
              color="secondary"
              endIcon={<RocketLaunch />}
              onClick={() => navigate('/spam')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              sx={{ px: 4, py: 1.5, fontSize: '1.05rem', borderRadius: 3 }}
            >
              Try the Live Demo
            </MotionButton>
          </MotionBox>
        </Container>
      </MotionBox>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <MotionTypography
          variant="h4"
          textAlign="center"
          sx={{ mb: 1 }}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.5 }}
        >
          What This Project Does
        </MotionTypography>
        <Typography variant="body1" textAlign="center" color="text.secondary" sx={{ mb: 6 }}>
          A full machine learning pipeline, from raw email text to a live, explainable prediction.
        </Typography>

        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={4} key={feature.title}>
              <MotionCard
                elevation={3}
                sx={{ height: '100%', p: 1, borderRadius: 3 }}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.5, delay: index * 0.15 }}
                whileHover={{ y: -8 }}
              >
                <CardContent sx={{ textAlign: 'center' }}>
                  <Box sx={{ color: 'primary.main', mb: 2 }}>{feature.icon}</Box>
                  <Typography variant="h6" sx={{ mb: 1.5 }}>{feature.title}</Typography>
                  <Typography variant="body2" color="text.secondary">{feature.description}</Typography>
                </CardContent>
              </MotionCard>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Tech Stack Section */}
      <Box sx={{ bgcolor: '#eceff1', py: 6 }}>
        <Container maxWidth="md" sx={{ textAlign: 'center' }}>
          <Typography variant="h5" sx={{ mb: 3 }}>Built With</Typography>
          <Stack direction="row" flexWrap="wrap" justifyContent="center" gap={1.5}>
            {techStack.map((tech, index) => (
              <MotionBox
                key={tech}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.35, delay: index * 0.05 }}
              >
                <Chip label={tech} color="primary" variant="outlined" sx={{ fontWeight: 500 }} />
              </MotionBox>
            ))}
          </Stack>
        </Container>
      </Box>

      {/* Closing CTA */}
      <Box sx={{ bgcolor: '#1a237e', color: '#fff', py: 8 }}>
        <Container maxWidth="sm" sx={{ textAlign: 'center' }}>
          <Security sx={{ fontSize: 48, mb: 2, opacity: 0.85 }} />
          <Typography variant="h5" sx={{ mb: 2 }}>Ready to see it classify a real email?</Typography>
          <MotionButton
            variant="outlined"
            color="inherit"
            size="large"
            endIcon={<Speed />}
            onClick={() => navigate('/spam')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            sx={{ borderRadius: 3, px: 4 }}
          >
            Open the Classifier
          </MotionButton>
        </Container>
      </Box>
    </Box>
  );
}
