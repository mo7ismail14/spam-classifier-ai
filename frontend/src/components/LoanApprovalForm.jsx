import { useState, useEffect } from 'react';
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
  Divider,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { CheckCircle, Cancel, Gavel, Analytics } from '@mui/icons-material';
import { AI_API_URL } from '../Utilti/constant';
import ConfusionMatrixCard from './ConfusionMatrixCard';
import GridSearchResultsTable from './GridSearchResultsTable';

const sampleOptionLabel = (sample) => {
  if (!sample) return '';
  const actual = sample.Loan_Status === 'Y' ? 'Approved' : 'Denied';
  return `Income ${sample.ApplicantIncome} | Credit History: ${sample.Credit_History === 1 ? 'good' : 'poor'} | ${sample.Property_Area} | Actual: ${actual}`;
};

export default function LoanApprovalForm() {
  const [fields, setFields] = useState([]);
  const [modelType, setModelType] = useState('');
  const [cvResults, setCvResults] = useState([]);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [formValues, setFormValues] = useState({});
  const [loadingSchema, setLoadingSchema] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSchema();
    fetchSamples();
  }, []);

  const fetchSchema = async () => {
    setLoadingSchema(true);
    setError('');
    try {
      const response = await axios.get(`${AI_API_URL}/loan/schema`);
      setFields(response.data.fields || []);
      setModelType(response.data.model_type || '');
      setCvResults(response.data.cv_results || []);
    } catch (err) {
      console.error(err);
      setError('Could not load the loan form. Make sure the AI service is running and trained.');
    } finally {
      setLoadingSchema(false);
    }
  };

  const fetchSamples = async () => {
    try {
      const response = await axios.get(`${AI_API_URL}/loan/samples`);
      setSamples(response.data || []);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFieldChange = (name, value) => {
    setFormValues((previous) => ({ ...previous, [name]: value }));
  };

  const handleSampleSelect = (event, sample) => {
    setSelectedSample(sample);
    setResult(null);
    setFormValues(sample ? { ...sample } : {});
  };

  const isFormComplete = fields.length > 0 && fields.every((field) => {
    const value = formValues[field.name];
    return value !== undefined && value !== null && value !== '';
  });

  const handleSubmit = async () => {
    setSubmitting(true);
    setError('');
    try {
      const response = await axios.post(`${AI_API_URL}/loan/predict`, formValues);
      setResult({
        ...response.data,
        actual: selectedSample ? selectedSample.Loan_Status : null
      });
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Failed to get a prediction from the backend server.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleClear = () => {
    setSelectedSample(null);
    setFormValues({});
    setResult(null);
  };

  const isApproved = result?.prediction === 'Y';
  const isPredictionCorrect = result?.actual ? result.actual === result.prediction : null;

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f0f2f5', pb: 6 }}>
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1a237e', mb: 3 }}>
          Loan Approval • Decision Tree Explainer
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Left Column: Applicant Form */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, borderRadius: 2 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#1a237e' }}>
                Applicant Details
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Load a sample applicant or fill in the details manually, then check eligibility.
              </Typography>

              {loadingSchema ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <CircularProgress size={24} />
                  <Typography variant="body2">Loading application form...</Typography>
                </Box>
              ) : (
                <>
                  <Autocomplete
                    options={samples}
                    getOptionLabel={sampleOptionLabel}
                    value={selectedSample}
                    onChange={handleSampleSelect}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Load a sample applicant"
                        placeholder="Type to filter sample applicants..."
                        variant="outlined"
                        fullWidth
                      />
                    )}
                    sx={{ mb: 3 }}
                  />

                  <Grid container spacing={2}>
                    {fields.map((field) => (
                      <Grid item xs={12} sm={6} key={field.name}>
                        {field.type === 'select' ? (
                          <FormControl fullWidth>
                            <InputLabel id={`${field.name}-label`}>{field.label}</InputLabel>
                            <Select
                              labelId={`${field.name}-label`}
                              label={field.label}
                              value={formValues[field.name] ?? ''}
                              onChange={(e) => handleFieldChange(field.name, e.target.value)}
                            >
                              {(field.options || []).map((option) => (
                                <MenuItem key={option} value={option}>{option}</MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                        ) : (
                          <TextField
                            fullWidth
                            type="number"
                            label={field.label}
                            value={formValues[field.name] ?? ''}
                            onChange={(e) => handleFieldChange(field.name, e.target.value)}
                          />
                        )}
                      </Grid>
                    ))}
                  </Grid>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 3 }}>
                    <Button variant="outlined" onClick={handleClear}>
                      Clear
                    </Button>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={handleSubmit}
                      disabled={submitting || !isFormComplete}
                      startIcon={submitting ? <CircularProgress size={20} color="inherit" /> : <Gavel />}
                      sx={{ px: 4, py: 1 }}
                    >
                      {submitting ? 'Evaluating...' : 'Check Eligibility'}
                    </Button>
                  </Box>
                </>
              )}
            </Paper>
          </Grid>

          {/* Right Column: Decision + Explanation + Metrics */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, borderRadius: 2, minHeight: '100%' }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#1a237e' }}>
                Decision Output
              </Typography>

              {result ? (
                <Box>
                  <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 2, border: '1px solid #e0e0e0', mb: 2 }}>
                    <Grid container spacing={2} alignItems="center">
                      {result.actual && (
                        <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary">ACTUAL OUTCOME</Typography>
                          <Box sx={{ mt: 0.5 }}>
                            <Chip
                              icon={result.actual === 'Y' ? <CheckCircle /> : <Cancel />}
                              label={result.actual === 'Y' ? 'APPROVED' : 'DENIED'}
                              color={result.actual === 'Y' ? 'success' : 'error'}
                              variant="filled"
                            />
                          </Box>
                        </Grid>
                      )}
                      <Grid item xs={result.actual ? 6 : 12}>
                        <Typography variant="caption" color="text.secondary">MODEL PREDICTION</Typography>
                        <Box sx={{ mt: 0.5 }}>
                          <Chip
                            icon={isApproved ? <CheckCircle /> : <Cancel />}
                            label={isApproved ? 'APPROVED' : 'DENIED'}
                            color={isApproved ? 'success' : 'error'}
                            sx={{ fontWeight: 'bold' }}
                          />
                        </Box>
                      </Grid>
                    </Grid>

                    {result.actual && (
                      <Box sx={{ mt: 2 }}>
                        {isPredictionCorrect ? (
                          <Alert severity="success" icon={<CheckCircle fontSize="inherit" />} sx={{ py: 0.5 }}>
                            <strong>Prediction Matches Actual Outcome!</strong>
                          </Alert>
                        ) : (
                          <Alert severity="warning" icon={<Cancel fontSize="inherit" />} sx={{ py: 0.5 }}>
                            <strong>Model Discrepancy (Misclassification)</strong>
                          </Alert>
                        )}
                      </Box>
                    )}

                    <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 2 }}>
                      AI Explanation
                    </Typography>
                    <Typography variant="body1" sx={{ mt: 0.5 }}>
                      {result.explanation}
                    </Typography>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  {result.metrics && (
                    <ConfusionMatrixCard
                      metrics={result.metrics}
                      negativeLabel="Denied"
                      positiveLabel="Approved"
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
                  <Typography variant="body1">No decision generated yet.</Typography>
                  <Typography variant="body2">Fill in the applicant details and click Check Eligibility.</Typography>
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Full Width: Grid Search Results */}
          {cvResults.length > 0 && (
            <Grid item xs={12}>
              <Paper elevation={2} sx={{ p: 3, borderRadius: 2 }}>
                <GridSearchResultsTable
                  modelType={modelType}
                  bestParams={cvResults.find((r) => r.rank === 1)?.params}
                  cvResults={cvResults}
                />
              </Paper>
            </Grid>
          )}
        </Grid>
      </Container>
    </Box>
  );
}
