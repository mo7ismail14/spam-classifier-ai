import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { Analytics } from '@mui/icons-material';

/**
 * Renders accuracy/precision/recall/F1 summary cards plus a 2x2 confusion
 * matrix table from a { tn, fp, fn, tp } metrics object. `negativeLabel` and
 * `positiveLabel` name the two classes (e.g. Ham/Spam, or Denied/Approved).
 */
export default function ConfusionMatrixCard({ metrics, negativeLabel = 'Negative', positiveLabel = 'Positive' }) {
  if (!metrics) return null;

  const { tp, tn, fp, fn } = metrics;
  const total = tp + tn + fp + fn;
  const accuracy = total > 0 ? ((tp + tn) / total) * 100 : 0;
  const precision = (tp + fp) > 0 ? (tp / (tp + fp)) * 100 : 0;
  const recall = (tp + fn) > 0 ? (tp / (tp + fn)) * 100 : 0;
  const f1 = (precision + recall) > 0 ? (2 * (precision * recall)) / (precision + recall) : 0;

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Analytics color="primary" /> Confusion Matrix & Performance Metrics
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <Card sx={{ bgcolor: '#e3f2fd', textAlign: 'center' }}>
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary" fontWeight="bold">ACCURACY</Typography>
              <Typography variant="h5" color="primary.main" fontWeight="bold">{accuracy.toFixed(1)}%</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card sx={{ bgcolor: '#e8f5e9', textAlign: 'center' }}>
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary" fontWeight="bold">PRECISION</Typography>
              <Typography variant="h5" color="success.main" fontWeight="bold">{precision.toFixed(1)}%</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card sx={{ bgcolor: '#fff3e0', textAlign: 'center' }}>
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary" fontWeight="bold">RECALL</Typography>
              <Typography variant="h5" color="warning.main" fontWeight="bold">{recall.toFixed(1)}%</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card sx={{ bgcolor: '#f3e5f5', textAlign: 'center' }}>
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary" fontWeight="bold">F1 SCORE</Typography>
              <Typography variant="h5" color="secondary.main" fontWeight="bold">{f1.toFixed(1)}%</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead sx={{ bgcolor: '#eceff1' }}>
            <TableRow>
              <TableCell align="center" colSpan={2} rowSpan={2} sx={{ fontWeight: 'bold', borderRight: '1px solid #ccc' }}>
                Total Test: {total}
              </TableCell>
              <TableCell align="center" colSpan={2} sx={{ fontWeight: 'bold' }}>
                Predicted Class
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell align="center" sx={{ fontWeight: 'bold', color: 'success.dark' }}>Predicted {negativeLabel}</TableCell>
              <TableCell align="center" sx={{ fontWeight: 'bold', color: 'error.dark' }}>Predicted {positiveLabel}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell rowSpan={2} sx={{ fontWeight: 'bold', writingMode: 'vertical-rl', transform: 'rotate(180deg)', textAlign: 'center', bgcolor: '#f7f9fa' }}>
                Actual Class
              </TableCell>
              <TableCell sx={{ fontWeight: 'bold', color: 'success.dark' }}>Actual {negativeLabel}</TableCell>
              <TableCell align="center" sx={{ bgcolor: '#e8f5e9', fontWeight: 'bold', color: 'success.dark' }}>
                <Typography variant="subtitle1" fontWeight="bold">{tn}</Typography>
                <Typography variant="caption">True Negative (TN)</Typography>
              </TableCell>
              <TableCell align="center" sx={{ bgcolor: '#ffebee', fontWeight: 'bold', color: 'error.dark' }}>
                <Typography variant="subtitle1" fontWeight="bold">{fp}</Typography>
                <Typography variant="caption">False Positive (FP)</Typography>
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 'bold', color: 'error.dark' }}>Actual {positiveLabel}</TableCell>
              <TableCell align="center" sx={{ bgcolor: '#ffebee', fontWeight: 'bold', color: 'error.dark' }}>
                <Typography variant="subtitle1" fontWeight="bold">{fn}</Typography>
                <Typography variant="caption">False Negative (FN)</Typography>
              </TableCell>
              <TableCell align="center" sx={{ bgcolor: '#e8f5e9', fontWeight: 'bold', color: 'success.dark' }}>
                <Typography variant="subtitle1" fontWeight="bold">{tp}</Typography>
                <Typography variant="caption">True Positive (TP)</Typography>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
