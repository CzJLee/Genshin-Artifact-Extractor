import artifact_extractor
import cv2
import pathlib

def test_img_hash(img_hash_algorithm = cv2.img_hash.pHash):
    # All images in this list should be the same.
    same_images = [
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1086.jpg",
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1087.jpg",
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1088.jpg",
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1089.jpg",
    ]

    # All images in this list should be different.
    different_images = [
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1090.jpg",
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1091.jpg",
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1092.jpg",
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1093.jpg",
        "/Users/Christian/Developer/Python/Genshin Artifact Extractor/_artifact_temp/video_output/1080.jpg",
    ]

    base_img = same_images[0]
    base_img_hash = artifact_extractor._get_img_hash(base_img, img_hash_algorithm=img_hash_algorithm)


    all_tests_passed = True
    print("Testing same images.")
    for img_path in same_images:
        img_hash = artifact_extractor._get_img_hash(img_path, img_hash_algorithm=img_hash_algorithm)
        if base_img_hash != img_hash:
            print(f"TEST FAILED: {base_img_hash} != {img_hash} for images {pathlib.Path(base_img).name} and {pathlib.Path(img_path).name}.")
            all_tests_passed = False

    print("Testing different images.")
    for img_path in different_images:
        img_hash = artifact_extractor._get_img_hash(img_path, img_hash_algorithm=img_hash_algorithm)
        if base_img_hash == img_hash:
            print(f"TEST FAILED: {base_img_hash} == {img_hash} for images {pathlib.Path(base_img).name} and {pathlib.Path(img_path).name}.")
            all_tests_passed = False
    
    if all_tests_passed:
        print("ALL TESTS PASSED.")


if __name__ == "__main__":
    print("pHash")
    test_img_hash(img_hash_algorithm=cv2.img_hash.pHash)

    print("averageHash")
    test_img_hash(img_hash_algorithm=cv2.img_hash.averageHash)

    print("blockMeanHash")
    test_img_hash(img_hash_algorithm=cv2.img_hash.blockMeanHash)

    print("colorMomentHash")
    test_img_hash(img_hash_algorithm=cv2.img_hash.colorMomentHash)

    print("marrHildrethHash")
    test_img_hash(img_hash_algorithm=cv2.img_hash.marrHildrethHash)
    
    print("radialVarianceHash")
    test_img_hash(img_hash_algorithm=cv2.img_hash.radialVarianceHash)