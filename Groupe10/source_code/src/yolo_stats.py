import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("\n--- Analyse TEST (depuis le JSON de Chahinez) ---")
json_path = 'reports/test_object_analysis.json'

try:
    with open(json_path, 'r') as f:
        test_data = json.load(f)
        
    # Chahinez a déjà tout calculé ! On récupère juste son dictionnaire
    print(f"Total d'images traitées : {test_data['processed_images']}")
    print(f"Total d'objets détectés : {test_data['total_objects']}")
    print(f"Moyenne d'objets par image : {test_data['average_objects_per_processed_image']}")
    
    # On récupère les fréquences des objets
    freq_dict = test_data['object_frequency']
    # On trie pour avoir le top 50
    top_50 = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    
    print("\nTop 5 des objets (TEST) :")
    for obj, count in top_50[:5]:
        print(f"  - {obj} : {count}")
        
    # Graphique
    labels, values = zip(*top_50)
    plt.figure(figsize=(15, 6))
    plt.bar(labels, values, color='orange')
    plt.xticks(rotation=90)
    plt.title('Top 50 Most Frequent Objects - TEST (YOLO Extraction)')
    plt.tight_layout()
    plt.savefig('graph1_top50_yolo_test.png')
    print("✅ Graphique sauvegardé : graph1_top50_yolo_test.png")

except FileNotFoundError:
    print("❌ Le fichier reports/test_object_analysis.json n'est pas trouvé.")