# AE-LSTM Embedding Generator

This project implements an LSTM Autoencoder for analyzing sequential data such as time-series or policy sequences. It includes two main components: 

1. **SequenceGenerator**: Prepares and pads sequences from a DataFrame.
2. **GeneralLSTMAutoencoder**: Builds, trains, and uses an LSTM autoencoder to extract embeddings and find nearest neighbors using FAISS.

**Key Steps**:
1. **Data Preparation**: Organize and pad sequences from your data.
2. **Model Training**: Train the LSTM autoencoder on these sequences.
3. **Embedding Extraction**: Use the trained model to compress sequences into embeddings.
4. **Nearest Neighbor Search**: Find similar sequences using these embeddings.


# Useage 
```python
from src.sequence_generator import SequenceGenerator
from src.ae_lstm import GeneralLSTMAutoencoder

# Generate dummy data (for demonstration)
# Generate sequences
sequence_generator = SequenceGenerator(df=dummy_df, id_column='policynumber', sequence_column='pol_exp_year', target_column='acpt_nt_ind')
padded_sequences, labels = sequence_generator.generate_sequences()

# Train the LSTM Autoencoder
lstm_autoencoder = GeneralLSTMAutoencoder(padded_sequences.shape[1], padded_sequences.shape[2])
history = lstm_autoencoder.fit(padded_sequences, epochs=50, batch_size=64)

# Extract embeddings
embeddings = lstm_autoencoder.encode(padded_sequences)

# Find nearest neighbors
distances, indices = lstm_autoencoder.find_nearest_neighbors(embeddings, embeddings, k=8)

```

# Data Format

### Dummy Data Generation

```python
np.random.seed(42)
n_policies = 100
max_exp_years = 5
n_features = 10
data = []
for policy in range(1, n_policies + 1):
    n_years = np.random.randint(1, max_exp_years + 1)
    for year in range(n_years):
        data.append([policy, year] + list(np.random.rand(n_features)) + [np.random.randint(0, 2)])
columns = ['policynumber', 'pol_exp_year'] + [f'feature_{i}' for i in range(n_features)] + ['acpt_nt_ind']
dummy_df = pd.DataFrame(data, columns=columns)

```


| policynumber | pol_exp_year | feature_0 | feature_1 | feature_2 | ... | feature_9 | acpt_nt_ind |
|--------------|--------------|-----------|-----------|-----------|-----|-----------|-------------|
| 1            | 0            | 0.374540  | 0.950714  | 0.731994  | ... | 0.708073  | 1           |
| 1            | 1            | 0.020584  | 0.969910  | 0.832443  | ... | 0.212339  | 1           |
| 1            | 2            | 0.181825  | 0.183405  | 0.304242  | ... | 0.524756  | 1           |
| 2            | 0            | 0.431945  | 0.291229  | 0.611853  | ... | 0.139494  | 0           |
| 2            | 1            | 0.292145  | 0.366362  | 0.456070  | ... | 0.199674  | 0           |
| 3            | 0            | 0.785176  | 0.199674  | 0.514234  | ... | 0.592415  | 1           |
| ...          | ...          | ...       | ...       | ...       | ... | ...       | ...         |
| 100          | 0            | 0.020584  | 0.969910  | 0.832443  | ... | 0.212339  | 0           |
| 100          | 1            | 0.431945  | 0.291229  | 0.611853  | ... | 0.139494  | 0           |
| 100          | 2            | 0.292145  | 0.366362  | 0.456070  | ... | 0.199674  | 1           |

This DataFrame structure shows multiple rows per policy, capturing the evolution of the policy over different experience years. Each policy number has sequential entries with corresponding features and the target variable.

# Applications

1. **Customer Similarity Analysis**:
   - **Example**: Match current policies with similar cancelled policies to identify patterns. For instance, you can analyze the mean values of certain features (e.g., premium amount, claim frequency) at which customers tend to cancel their policies. This can help in understanding risk factors and improving customer retention strategies.

2. **Fraud Detection**:
   - **Example**: Identify sequences of activities that are similar to known fraudulent behaviors. By comparing the embeddings of new policy sequences with those of previously identified fraudulent cases, it is possible to flag suspicious activities early.

3. **Churn Prediction**:
   - **Example**: Match current active policies with similar policies that have churned in the past. By analyzing the sequences leading up to the churn, you can develop predictive models to forecast which current policies are at risk of churning, enabling proactive customer engagement.

4. **Claim Prediction**:
   - **Example**: Identify policies with sequences similar to those that had high claim amounts. By analyzing the life cycle of these policies, you can predict which current policies are likely to result in high claims, allowing for better risk management and resource allocation.

5. **Lifecycle Analysis**:
   - **Example**: Compare the entire lifecycle of a policy with other policies to understand typical patterns and deviations. This can be used to identify outliers or unusual patterns that might indicate underlying issues or opportunities for product improvements.

### Why Traditional Methods Might Fail

Traditional methods such as Generalized Linear Models (GLM) and XGBoost often assume that observations are independently and identically distributed (IID). This assumption is violated in the context of sequential data, where multiple rows per policy capture the evolving nature of the policy over time. Specifically, each observation is not independent of the previous ones, and the temporal dependencies are crucial for understanding the entire lifecycle of a policy.

**Additional Reasons Why Traditional Methods Might Fail:**

1. **Ignoring Temporal Dependencies**:
   - Traditional models like GLM and XGBoost treat each row as independent, which fails to capture the temporal dependencies and sequence patterns inherent in policy data. This can lead to suboptimal predictions and missed insights that are critical for understanding trends over time.

2. **Inability to Model Sequential Patterns**:
   - Sequential data often exhibit patterns that unfold over time. Traditional methods might fail to capture these sequential patterns, such as the progression of claim amounts or the likelihood of policy renewal. LSTM Autoencoders are specifically designed to handle such sequential dependencies.

3. **Feature Engineering Complexity**:
   - With traditional methods, capturing temporal dynamics often requires extensive feature engineering to create lagged features or aggregations over time windows. This process is not only labor-intensive but may also miss important patterns. LSTM Autoencoders can learn these patterns directly from the raw sequential data.

4. **Handling Missing Data**:
   - Sequential data can have varying lengths and missing values. Traditional methods struggle with such data irregularities. LSTM Autoencoders, especially with masking layers, can handle sequences of varying lengths and deal with missing data more effectively.

5. **Scalability with Sequence Length**:
   - As the length of the sequences increases, traditional methods can become computationally expensive and less effective. LSTM Autoencoders are designed to scale with sequence length, making them more suitable for long sequences.

By leveraging LSTM Autoencoders, you can overcome these limitations and achieve a more comprehensive and accurate analysis of sequential  data, capturing the full lifecycle and dependencies inherent in the data.
