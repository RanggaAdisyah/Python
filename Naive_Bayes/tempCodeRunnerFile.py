# Data Training
data_training = [
    
    {'Nama': 'Genshin Impact', 'Grafik': 8, 'Gameplay': 9, 'Kisah': 7, 'Rating': 'Baik'},
    {'Nama': 'Assasins Creed: Black Flag ', 'Grafik': 7, 'Gameplay': 8, 'Kisah': 9, 'Rating': 'Baik'},
    {'Nama': 'Plants Vs Zombie ', 'Grafik': 6, 'Gameplay': 7, 'Kisah': 8, 'Rating': 'Baik'},
    {'Nama': 'Feeding Frenzy ', 'Grafik': 4, 'Gameplay': 5, 'Kisah': 3, 'Rating': 'Buruk'},
    {'Nama': 'Marvel Spider-Man: Miles Moralas ', 'Grafik': 9, 'Gameplay': 9, 'Kisah': 7, 'Rating': 'Normal'},
    {'Nama': 'EA: FC 2024', 'Grafik': 9, 'Gameplay': 8, 'Kisah': 5, 'Rating': 'Baik'},
    {'Nama': 'No Man Sky ', 'Grafik': 6, 'Gameplay': 4, 'Kisah': 3, 'Rating': 'Buruk'},
    {'Nama': 'The Witcher 3 ', 'Grafik': 7, 'Gameplay': 9, 'Kisah': 10, 'Rating': 'Baik'},
    {'Nama': 'Clash Royale ', 'Grafik': 5, 'Gameplay': 7, 'Kisah': 6, 'Rating': 'Normal'},
    {'Nama': 'Among Us ', 'Grafik': 3, 'Gameplay': 5, 'Kisah': 3, 'Rating': 'Buruk'},
    {'Nama': 'Space of the Unbound ', 'Grafik': 5, 'Gameplay': 6, 'Kisah': 9, 'Rating': 'Normal'},
    {'Nama': 'Tower of Fantasy ', 'Grafik': 6, 'Gameplay': 5, 'Kisah': 3, 'Rating': 'Buruk'},
        
    # {'Nama': 'Fantasy World', 'Grafik': 5, 'Gameplay': 6, 'Kisah': 8, 'Rating': 'Baik'},
    # {'Nama': 'Galaxy Explorer', 'Grafik': 9, 'Gameplay': 8, 'Kisah': 9, 'Rating': 'Baik'},
    # {'Nama': 'Kingdom Defense', 'Grafik': 6, 'Gameplay': 7, 'Kisah': 5, 'Rating': 'Normal'},
    # {'Nama': 'Heroic Saga', 'Grafik': 7, 'Gameplay': 8, 'Kisah': 6, 'Rating': 'Baik'},
    # {'Nama': 'Space Odyssey', 'Grafik': 4, 'Gameplay': 5, 'Kisah': 6, 'Rating': 'Normal'},
    # {'Nama': 'Ultimate Battle', 'Grafik': 10, 'Gameplay': 9, 'Kisah': 10, 'Rating': 'Baik'},
    # {'Nama': 'Mystic Lands', 'Grafik': 3, 'Gameplay': 4, 'Kisah': 4, 'Rating': 'Buruk'},
    # {'Nama': 'Legendary Quest', 'Grafik': 8, 'Gameplay': 8, 'Kisah': 8, 'Rating': 'Baik'},
    # {'Nama': 'Dark Realms', 'Grafik': 6, 'Gameplay': 6, 'Kisah': 7, 'Rating': 'Baik'}
    
]

# Calculate prior probabilities
def calculate_prior_probabilities(training_data):
    total_count = len(training_data)
    rating_counts = {'Buruk': 0, 'Normal': 0, 'Baik': 0}
    
    for data in training_data:
        rating_counts[data['Rating']] += 1

    priors = {rating: count / total_count for rating, count in rating_counts.items()}
    return priors

# Calculate likelihood probabilities with Laplace smoothing
def calculate_likelihood_probabilities(training_data):
    likelihoods = {
        'Buruk': {'Grafik': {}, 'Gameplay': {}, 'Kisah': {}},
        'Normal': {'Grafik': {}, 'Gameplay': {}, 'Kisah': {}},
        'Baik': {'Grafik': {}, 'Gameplay': {}, 'Kisah': {}}
    }
    feature_values = {'Grafik': [], 'Gameplay': [], 'Kisah': []}

    for data in training_data:
        rating = data['Rating']
        for feature in ['Grafik', 'Gameplay', 'Kisah']:
            value = data[feature]
            feature_values[feature].append(value)
            if value not in likelihoods[rating][feature]:
                likelihoods[rating][feature][value] = 0
            likelihoods[rating][feature][value] += 1
    
    for rating in likelihoods:
        for feature in likelihoods[rating]:
            total_values = sum(likelihoods[rating][feature].values())
            unique_values = len(set(feature_values[feature]))
            for value in likelihoods[rating][feature]:
                likelihoods[rating][feature][value] = (likelihoods[rating][feature][value] + 1) / (total_values + unique_values)
            # Add smoothing for unseen values
            likelihoods[rating][feature]['<UNK>'] = 1 / (total_values + unique_values)
    
    return likelihoods

# Classify new data using Naive Bayes
def classify_naive_bayes(grafik, gameplay, kisah, priors, likelihoods):
    probabilities = {rating: priors[rating] for rating in priors}
    
    for rating in probabilities:
        for feature, value in zip(['Grafik', 'Gameplay', 'Kisah'], [grafik, gameplay, kisah]):
            if value in likelihoods[rating][feature]:
                probabilities[rating] *= likelihoods[rating][feature][value]
            else:
                probabilities[rating] *= likelihoods[rating][feature]['<UNK>']
    
    return max(probabilities, key=probabilities.get)

# Prepare the model
priors = calculate_prior_probabilities(data_training)
likelihoods = calculate_likelihood_probabilities(data_training)

# Function to test with input values
def test_game_rating_naive_bayes():
    grafik = int(input("Masukkan nilai Grafik (1-10): "))
    gameplay = int(input("Masukkan nilai Gameplay (1-10): "))
    kisah = int(input("Masukkan nilai Kisah/Cerita (1-10): "))

    rating = classify_naive_bayes(grafik, gameplay, kisah, priors, likelihoods)
    print(f'Rating game berdasarkan input adalah: {rating}')

# Test the function
test_game_rating_naive_bayes()