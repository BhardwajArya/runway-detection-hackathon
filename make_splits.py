import glob, random, csv, os

# Collect all image filenames from train folder
imgs = sorted(glob.glob('images/train/*.png') + glob.glob('images/train/*.jpg'))
imgs = [os.path.basename(p) for p in imgs]

random.seed(42)
random.shuffle(imgs)

n = len(imgs)
train = imgs[:int(0.8*n)]
val   = imgs[int(0.8*n):int(0.9*n)]
test  = imgs[int(0.9*n):]

def write_csv(name, arr):
    with open(name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename'])
        for fn in arr:
            w.writerow([fn])

write_csv('train_files.csv', train)
write_csv('val_files.csv', val)
write_csv('test_files.csv', test)

print("Split sizes -> train:", len(train), "val:", len(val), "test:", len(test))
