import csv

csv_file1 = 'book_person.csv'
csv_file2 = '_book_person.csv'

lines = []

with open(csv_file1,'r',newline="",encoding='utf-8') as file_in:
	reader = csv.reader(file_in)
	for line in reader:
		if line:
			lines.append(line)

with open(csv_file2,'w',newline="",encoding='utf-8') as file_out:
	writer = csv.writer(file_out)
	for line in lines:
		writer.writerow(line[:-4])


			
