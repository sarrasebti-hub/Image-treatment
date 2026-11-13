
import tempfile
from pathlib import Path

import extract_yolo_10000 as extraction


def main():

    # Utiliser un fichier temporaire indépendant
    with tempfile.TemporaryDirectory() as temp_dir:

        test_file = Path(temp_dir) / "checkpoint_test.json"

        extraction.CHECKPOINT_FILE = test_file

        # Simuler deux images déjà traitées
        sample_data = [
            {
                "image_id": 100001,
                "file_name": "000000100001.jpg",
                "detections": []
            },
            {
                "image_id": 100002,
                "file_name": "000000100002.jpg",
                "detections": []
            }
        ]

        # Tester la sauvegarde
        extraction.save_json(sample_data, test_file)

        # Tester le chargement
        loaded_data = extraction.load_checkpoint()

        assert loaded_data == sample_data

        # Tester la logique de reprise
        all_ids = [100001, 100002, 100003, 100004]

        completed_ids = {
            item["image_id"] for item in loaded_data
        }

        remaining_ids = [
            image_id for image_id in all_ids
            if image_id not in completed_ids
        ]

        assert remaining_ids == [100003, 100004]

        print("Sauvegarde : OK")
        print("Chargement : OK")
        print("Identification des images traitees : OK")
        print("Reprise des images restantes : OK")
        print("Fichier temporaire supprime automatiquement.")


if __name__ == "__main__":
    main()
