import numpy as np
import matplotlib.pyplot as plt

file_path = "Dataset"
dts_folder = "MSL"
dts_name = "MSL_test.npy"
dts_url = file_path + "/" + dts_folder + "/" + dts_name
chunk_size = 2500
data_Type = object

arr_np = np.load(dts_url)
print(f"Count: {len(arr_np)}")

print("Shape: ", arr_np.shape)
print("Type: ", type(arr_np))
print("Print data: ", arr_np[:100])

if arr_np.dtype.names:
    print("Column names: ", arr_np.dtype.names)
else:
    print("The data does not have column names.")

# first_column  = arr_np[:, 0]
# second_column = arr_np[:, 1]
# third_column  = arr_np[:, 2]
# forth_column  = arr_np[:, 3]
# fifth_column  = arr_np[:, 4]
# sixth_column  = arr_np[:, 5]
# 
# plt.plot(range(len(first_column)), first_column, linestyle='-', color='blue', label="First Column")
# plt.plot(range(len(second_column)), second_column, linestyle='--', color='green', label="Second Column")
# plt.plot(range(len(third_column)), third_column, linestyle=':', color='red', label="Third Column")
# plt.plot(range(len(forth_column)), third_column, linestyle=':', color='red', label="Forth Column")
# plt.plot(range(len(fifth_column)), third_column, linestyle=':', color='red', label="Fifth Column")
# plt.plot(range(len(sixth_column)), third_column, linestyle=':', color='red', label="Sixth Column")

# plt.title("MSL Data")
# plt.xlabel("")
# plt.ylabel("")
# plt.grid(True)
# plt.show()

np.array_split(arr_np, 20)
head_arr = arr_np[:chunk_size]
tail_arr = arr_np[10:]
#print(head_arr)
#print(tail_arr)
arr_np5 = head_arr
arr_np5 = np.array(arr_np5, dtype=data_Type)
print(f"Count: {len(arr_np5)}")
# print(f"Count: {(arr_np5.itemsize)}")

np.save(file_path + "/" + dts_folder + "/MSL_" + str(chunk_size) + ".npy", arr_np5 )
arr_np5.tofile(file_path + "/" + dts_folder + "/MSL_" + str(chunk_size) + ".csv", sep=',')
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
print(newarr[1])
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
newarr = np.array_split(arr, 3)
print(newarr)