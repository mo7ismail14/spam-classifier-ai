import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Typography,
  Container,
  Grid,
  Paper,
  Box,
  Autocomplete,
  TextField,
  Button,
  Chip,
  LinearProgress,
  Divider,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  MarkEmailRead,
  ReportProblem,
  CheckCircle,
  Cancel,
  Analytics,
  Speed
} from '@mui/icons-material';
import { AI_API_URL } from '../Utilti/constant';
import ConfusionMatrixCard from './ConfusionMatrixCard';


export default function SpamHam() {
  const [emails, setEmails] = useState([]);
  const [selectedEmail, setSelectedEmail] = useState(null);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [fetchingEmails, setFetchingEmails] = useState(true);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState('');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  // Fetch sample emails and available models on load
  useEffect(() => {
    fetchSampleEmails();
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${AI_API_URL}/models`);
      const available = response.data.models || [];
      setModels(available);
      if (available.length > 0) {
        setSelectedModel(available[0]);
      }
    } catch (err) {
      console.error(err);
      setError('Could not load available models from backend.');
    }
  };

  const fetchSampleEmails = async () => {
    setFetchingEmails(true);
    setError('');
    try {
      const response = await axios.get(`${AI_API_URL}/emails`);
      setEmails(response.data);
    } catch (err) {
      console.error(err);
      setError('Could not connect to backend or model not ready. Make sure backend and AI services are running.');
    } finally {
      setFetchingEmails(false);
    }
  };

  const handleEmailSelect = (event, value) => {
    setSelectedEmail(value);
    setPredictionResult(null);
    setInputText(value ? (value.Message || '') : '');
  };

  const handlePredict = async () => {
    const text = inputText;
    const actualLabel = selectedEmail ? selectedEmail['Spam/Ham'] : null;

    if (!text.trim()) {
      setError('Please select an email or enter text to classify.');
      return;
    }
    if (!selectedModel) {
      setError('Please choose a model before classifying.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const response = await axios.post(`${AI_API_URL}/predict`, { text, model: selectedModel });
      setPredictionResult({
        ...response.data,
        actual: actualLabel
      });
    } catch (err) {
      console.error(err);
      setError('Failed to get prediction from backend server.');
    } finally {
      setLoading(false);
    }
  };

  // Calculate metrics from confusion matrix
  const isPredictionCorrect = predictionResult && predictionResult.actual
    ? predictionResult.actual.toLowerCase() === predictionResult.prediction.toLowerCase()
    : null;

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f0f2f5', pb: 6 }}>
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1a237e', mb: 3 }}>
          Spam Classification • Multi-Model Microservice
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Left Column: Email Selection and Input */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, borderRadius: 2 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#1a237e' }}>
                Select or Input Email Text
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Pick a sample email using the Autocomplete dropdown or paste custom email content, choose a model, then classify.
              </Typography>

              {fetchingEmails ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <CircularProgress size={24} />
                  <Typography variant="body2">Loading sample emails from dataset...</Typography>
                </Box>
              ) : (
                <Autocomplete
                  options={emails}
                  getOptionLabel={(option) => {
                    const snippet = (option.Message || '').replace(/\s+/g, ' ').slice(0, 75);
                    return `${snippet}...`;
                  }}
                  value={selectedEmail}
                  onChange={handleEmailSelect}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      label="Select a sample email"
                      placeholder="Type to filter emails..."
                      variant="outlined"
                      fullWidth
                    />
                  )}
                  sx={{ mb: 2 }}
                />
              )}

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="model-select-label">Model</InputLabel>
                <Select
                  labelId="model-select-label"
                  label="Model"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {models.map((name) => (
                    <MenuItem key={name} value={name}>{name}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <TextField
                label="Email Text Content"
                multiline
                rows={10}
                fullWidth
                variant="outlined"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Select an email above or type your own email message text here..."
                sx={{ mb: 2 }}
              />

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Button
                  variant="outlined"
                  onClick={() => {
                    setSelectedEmail(null);
                    setInputText('');
                    setPredictionResult(null);
                  }}
                >
                  Clear
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => handlePredict()}
                  disabled={loading || !inputText.trim() || !selectedModel}
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <Speed />}
                  sx={{ px: 4, py: 1 }}
                >
                  {loading ? `Analyzing with ${selectedModel}...` : 'Classify Email'}
                </Button>
              </Box>
            </Paper>
          </Grid>

          {/* Right Column: Prediction Details and Metrics Dashboard */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, borderRadius: 2, minHeight: '100%' }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#1a237e' }}>
                Classification Output
              </Typography>

              {predictionResult ? (
                <Box>
                  {/* Actual vs Predicted Comparison */}
                  <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 2, border: '1px solid #e0e0e0', mb: 2 }}>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">ACTUAL LABEL</Typography>
                        <Box sx={{ mt: 0.5 }}>
                          {predictionResult.actual ? (
                            <Chip
                              icon={predictionResult.actual.toLowerCase() === 'spam' ? <ReportProblem /> : <MarkEmailRead />}
                              label={predictionResult.actual.toUpperCase()}
                              color={predictionResult.actual.toLowerCase() === 'spam' ? 'error' : 'success'}
                              variant="filled"
                            />
                          ) : (
                            <Chip label="User Input" variant="outlined" />
                          )}
                        </Box>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          {(predictionResult.model || 'MODEL').toUpperCase()} PREDICTION
                        </Typography>
                        <Box sx={{ mt: 0.5 }}>
                          <Chip
                            icon={predictionResult.prediction === 'spam' ? <ReportProblem /> : <MarkEmailRead />}
                            label={predictionResult.prediction.toUpperCase()}
                            color={predictionResult.prediction === 'spam' ? 'error' : 'success'}
                          />
                        </Box>
                      </Grid>
                    </Grid>

                    {predictionResult.actual && (
                      <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                        {isPredictionCorrect ? (
                          <Alert severity="success" icon={<CheckCircle fontSize="inherit" />} sx={{ width: '100%', py: 0.5 }}>
                            <strong>Prediction Matches Actual Label!</strong>
                          </Alert>
                        ) : (
                          <Alert severity="warning" icon={<Cancel fontSize="inherit" />} sx={{ width: '100%', py: 0.5 }}>
                            <strong>Model Discrepancy (Misclassification)</strong>
                          </Alert>
                        )}
                      </Box>
                    )}

                    {/* Confidence Meter */}
                    <Box sx={{ mt: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2" color="text.secondary">Model Confidence</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {(predictionResult.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={predictionResult.confidence * 100}
                        color={predictionResult.prediction === 'spam' ? 'error' : 'success'}
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                    </Box>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  {/* Confusion Matrix Dashboard */}
                  {predictionResult.metrics && (
                    <ConfusionMatrixCard
                      metrics={predictionResult.metrics}
                      negativeLabel="Ham"
                      positiveLabel="Spam"
                    />
                  )}
                </Box>
              ) : (
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    py: 8,
                    color: 'text.secondary',
                    textAlign: 'center'
                  }}
                >
                  <Analytics sx={{ fontSize: 60, mb: 1, opacity: 0.3 }} />
                  <Typography variant="body1">No prediction generated yet.</Typography>
                  <Typography variant="body2">Select an email or enter text and click Classify.</Typography>
                </Box>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}
