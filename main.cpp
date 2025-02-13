#include <iostream>
#include <fstream>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <cmath>
#include <algorithm> // For max_element

using namespace std;

// Tokenization: Read words from a file and split into tokens
vector<string> tokenizeText(const string& filename) {
    ifstream file(filename, ios::in);
    vector<string> tokenWords;

    if (!file) {
        cerr << "Error: File could not be opened!" << endl;
        return {};
    }

    string word;
    while (file >> word) {
        string currentToken;
        for (char c : word) {
            if (ispunct(c)) {
                if (!currentToken.empty()) {
                    tokenWords.push_back(currentToken);
                    currentToken.clear();
                }
            } else {
                currentToken += tolower(c);
            }
        }
        if (!currentToken.empty()) {
            tokenWords.push_back(currentToken);
        }
    }

    file.close();
    return tokenWords;
}

// Build vocabulary and assign indices
unordered_map<string, int> buildVocabulary(const vector<string>& words) {
    unordered_map<string, int> vocab;
    int index = 0;
    for (const string& word : words) {
        if (!vocab.count(word)) {
            vocab[word] = index++;
        }
    }
    return vocab;
}

// Generate context-target pairs (CBOW)
vector<pair<vector<int>, int>> createContextTargetPairsIndexed(const vector<string>& tokens, int window_size, const unordered_map<string, int>& vocab) {
    vector<pair<vector<int>, int>> context_target_pairs;
    int n = tokens.size();

    for (int i = 0; i < n; ++i) {  // Corrected loop bounds
        if (i < window_size || i >= n - window_size) continue;
        vector<int> context;
        for (int j = i - window_size; j <= i + window_size; ++j) {
            if (j != i) {
                context.push_back(vocab.at(tokens[j]));
            }
        }
        int target = vocab.at(tokens[i]);
        context_target_pairs.push_back(make_pair(context, target));
    }
    return context_target_pairs;
}

// Initialize weight matrices
pair<vector<vector<double>>, vector<vector<double>>> initializeWeights(int vocab_size, int embedding_dim) {
    vector<vector<double>> W1(vocab_size, vector<double>(embedding_dim));
    vector<vector<double>> W2(embedding_dim, vector<double>(vocab_size));

    srand(time(0));

    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            W1[i][j] = ((double) rand() / RAND_MAX - 0.5) / embedding_dim;
        }
    }

    for (int i = 0; i < embedding_dim; ++i) {
        for (int j = 0; j < vocab_size; ++j) {
            W2[i][j] = ((double) rand() / RAND_MAX - 0.5) / embedding_dim;
        }
    }
    return {W1, W2};
}

// Softmax function with numerical stability
vector<double> softmax(const vector<double>& scores) {
    vector<double> exp_scores(scores.size());
    double max_score = *max_element(scores.begin(), scores.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < scores.size(); i++) {
        exp_scores[i] = exp(scores[i] - max_score);
        sum_exp += exp_scores[i];
    }
    for (double& val : exp_scores) {
        val /= sum_exp;
    }
    return exp_scores;
}

// Forward pass returning probabilities and hidden layer
pair<vector<double>, vector<double>> forwardPass(const vector<int>& context_indices, 
    const vector<vector<double>>& W1, const vector<vector<double>>& W2) {
    
    int embedding_dim = W1[0].size();
    int vocab_size = W2[0].size();
    
    // Average context word embeddings
    vector<double> hidden(embedding_dim, 0.0);
    for (int idx : context_indices) {
        for (int j = 0; j < embedding_dim; j++) {
            hidden[j] += W1[idx][j];
        }
    }
    
    // Normalize by context window size
    double scale = 1.0 / context_indices.size();
    for (double& val : hidden) {
        val *= scale;
    }

    // Output layer
    vector<double> output_scores(vocab_size, 0.0);
    for (int j = 0; j < vocab_size; j++) {
        double score = 0.0;
        for (int k = 0; k < embedding_dim; k++) {
            score += hidden[k] * W2[k][j];
        }
        output_scores[j] = score;
    }

    return {softmax(output_scores), hidden};
}

// Train the model using gradient descent
void trainModel(vector<pair<vector<int>, int>>& context_target_pairs, 
    vector<vector<double>>& W1, vector<vector<double>>& W2, 
    int epochs, double learning_rate) {
    
    int embedding_dim = W1[0].size();
    int vocab_size = W2[0].size();

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for (const auto& pair : context_target_pairs) {
            const vector<int>& context_indices = pair.first;
            int target_idx = pair.second;

            // Forward pass
            auto [probs, hidden] = forwardPass(context_indices, W1, W2);
            
            // Cross-entropy loss
            total_loss -= log(max(probs[target_idx], 1e-10));

            // Gradient for output layer
            vector<double> output_grad = probs;
            output_grad[target_idx] -= 1.0;  // d(cross_entropy)/d(logits)

            // Gradient for hidden layer
            vector<double> hidden_grad(embedding_dim, 0.0);
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < vocab_size; k++) {
                    hidden_grad[j] += output_grad[k] * W2[j][k];
                }
            }

            // Update W2 (output embeddings)
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < vocab_size; k++) {
                    W2[j][k] -= learning_rate * output_grad[k] * hidden[j];
                }
            }

            // Update W1 (input embeddings)
            double context_scale = learning_rate / context_indices.size();
            for (int idx : context_indices) {
                for (int j = 0; j < embedding_dim; j++) {
                    W1[idx][j] -= context_scale * hidden_grad[j];
                }
            }
        }

        cout << "Epoch " << epoch + 1 << " Loss: " 
             << total_loss / context_target_pairs.size() << endl;
    }
}

// Extract word embeddings
vector<double> getWordEmbedding(const string& word, const unordered_map<string, int>& vocab, const vector<vector<double>>& W1) {
    vector<double> embedding(W1[0].size(), 0.0);

    if (vocab.find(word) != vocab.end()) {
        int idx = vocab.at(word);
        return W1[idx];
    } else {
        cout << "Word not in vocabulary!" << endl;
        return embedding;
    }
}

// Main function
// Main function
int main() {
    cout << "Program started!" << endl;
    string filename = "C:\\word2vecfromscratch\\001ssb.txt"; // Change to your text file
    
    // Tokenization step
    vector<string> tokens = tokenizeText(filename);
    if (tokens.empty()) {
        cerr << "No tokens processed. Check input file." << endl;
        return 1;
    }

    // Display first 20 tokens for verification
    cout << "\nFirst 20 tokens:" << endl;
    for(int i = 0; i < 20 && i < tokens.size(); i++) {
        cout << "[" << i << "] " << tokens[i] << endl;
    }
    cout << "Total tokens: " << tokens.size() << "\n" << endl;

    // Rest of the existing code
    unordered_map<string, int> vocab = buildVocabulary(tokens);
    int window_size = 2;
    int vocab_size = vocab.size();
    int embedding_dim = 128;

    if (vocab_size == 0) {
        cerr << "Vocabulary is empty." << endl;
        return 1;
    }
    else{
        cout<<"all good1\n";
    }

    auto [W1, W2] = initializeWeights(vocab_size, embedding_dim);

    vector<pair<vector<int>, int>> context_target_pairs = createContextTargetPairsIndexed(tokens, window_size, vocab);
    if (context_target_pairs.empty()) {
        cerr << "No context-target pairs generated. Check window size and input text." << endl;
        return 1;
    }
    else{
        cout<<"all good2";
    }

    int epochs = 10;
    double learning_rate = 100000;
    trainModel(context_target_pairs, W1, W2, epochs, learning_rate);

    string test_word = "the";
    vector<double> embedding = getWordEmbedding(test_word, vocab, W1);
    cout << "Embedding for '" << test_word << "': ";
    for (double val : embedding) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}