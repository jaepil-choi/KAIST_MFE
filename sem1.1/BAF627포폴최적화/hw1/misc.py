# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
# ---

# %%
## Sum = 1이 되는 (-1, 1) 사이의 weights 생성

def generate_random_weights(size):
    # Step 1: Generate random weights with values between -1 and 1
    weights = np.random.uniform(-1, 1, size)
    
    # Step 2: Compute the sum of the absolute values of these weights
    abs_sum = np.sum(np.abs(weights))
    
    # Step 3: Normalize the weights by dividing each weight by the sum of the absolute values
    normalized_weights = weights / abs_sum
    
    # Ensure the sum of weights is exactly 1
    normalized_weights = normalized_weights / np.sum(normalized_weights)
    
    return normalized_weights

# Example usage
weights = generate_random_weights(1000)
negative_weights_count = (weights < 0).sum()
weights_sum = np.sum(weights)
weights[:10], negative_weights_count, weights_sum

