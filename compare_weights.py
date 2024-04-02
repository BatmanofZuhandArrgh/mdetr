def compare_files(file1_path, file2_path):
    # Open both files in binary mode
    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
        # Read the contents of both files
        content1 = file1.read()
        content2 = file2.read()

        # Compare the contents
        if content1 == content2:
            print("Files are the same.")
        else:
            print("Files are different.")

# Example usage
file1_path = "weights/flickr30k_ft_pretrained_EB3_checkpoint.pth"
file2_path = "weights/pretrained_EB3_checkpoint.pth"
compare_files(file1_path, file2_path)
