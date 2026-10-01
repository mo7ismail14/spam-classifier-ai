import { useState } from 'react';
import {
  Box,
  Typography,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import { Tune, CheckCircle } from '@mui/icons-material';

const DEFAULT_VISIBLE_ROWS = 10;

const formatParamValue = (value) => (value === null ? 'None' : String(value));

/**
 * Shows the winning hyperparameters as chips, plus a table of every combination
 * GridSearchCV tried (best-first), with the winning row(s) marked with a checkmark.
 */
export default function GridSearchResultsTable({ modelType, bestParams, cvResults }) {
  const [showAll, setShowAll] = useState(false);

  if (!cvResults || cvResults.length === 0) return null;

  const paramNames = Object.keys(cvResults[0].params);
  const visibleResults = showAll ? cvResults : cvResults.slice(0, DEFAULT_VISIBLE_ROWS);

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Tune color="primary" /> {modelType || 'Model'} Hyperparameter Search
      </Typography>

      {bestParams && (
        <Box sx={{ mb: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {Object.entries(bestParams).map(([key, value]) => (
            <Chip
              key={key}
              label={`${key}: ${formatParamValue(value)}`}
              color="success"
              variant="outlined"
              icon={<CheckCircle />}
            />
          ))}
        </Box>
      )}

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead sx={{ bgcolor: '#eceff1' }}>
            <TableRow>
              <TableCell align="center" sx={{ fontWeight: 'bold' }}>Best</TableCell>
              <TableCell sx={{ fontWeight: 'bold' }}>Rank</TableCell>
              {paramNames.map((name) => (
                <TableCell key={name} sx={{ fontWeight: 'bold' }}>{name}</TableCell>
              ))}
              <TableCell align="right" sx={{ fontWeight: 'bold' }}>Mean CV F1 Score</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {visibleResults.map((result, index) => {
              const isBest = result.rank === 1;
              return (
                <TableRow
                  key={index}
                  sx={isBest ? { bgcolor: '#e8f5e9' } : undefined}
                >
                  <TableCell align="center">
                    {isBest && <CheckCircle color="success" fontSize="small" />}
                  </TableCell>
                  <TableCell>{result.rank}</TableCell>
                  {paramNames.map((name) => (
                    <TableCell key={name}>{formatParamValue(result.params[name])}</TableCell>
                  ))}
                  <TableCell align="right">{result.mean_test_score.toFixed(4)}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {cvResults.length > DEFAULT_VISIBLE_ROWS && (
        <Button size="small" onClick={() => setShowAll((prev) => !prev)} sx={{ mt: 1 }}>
          {showAll ? 'Show fewer results' : `Show all ${cvResults.length} combinations tried`}
        </Button>
      )}
    </Box>
  );
}
