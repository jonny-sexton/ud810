# Problem Set 5: Optic Flow

## Question 1

### A

`Shift0.png` and `ShiftR2.png` optical flow:

![ps5-1-a-1.png](output/ps5-1-a-1.png)

`Shift0.png` and `ShiftR2U5.png` optical flow:

![ps5-1-a-2.png](output/ps5-1-a-2.png)

I had to weigh the kernel with a gaussian and increase the size of the window to avoid the "aperture" effect.

### B

`Shift0.png` and `ShiftR10.png` optical flow:

![ps5-1-b-1.png](output/ps5-1-b-1.png)

`Shift0.png` and `ShiftR20.png` optical flow:

![ps5-1-b-2.png](output/ps5-1-b-2.png)

`Shift0.png` and `ShiftR40.png` optical flow:

![ps5-1-b-3.png](output/ps5-1-b-3.png)

The algorithm falls apart the bigger the displacement.This is because LK cannot handle larger displacements.

## Question 2

### A

Gaussian pyramid:

![ps5-2-a-1.png](output/ps5-2-a-1.png)

### B

Laplacian pyramid:

![ps5-2-b-1.png](output/ps5-2-b-1.png)

## Question 3

### A

`DataSeq1` optical flow:

![ps5-3-a-1.png](output/ps5-3-a-1.png)

`DataSeq1` difference:

![ps5-3-a-2.png](output/ps5-3-a-2.png)

`DataSeq2` optical flow:

![ps5-3-a-3.png](output/ps5-3-a-3.png)

`DataSeq2` difference:

![ps5-3-a-4.png](output/ps5-3-a-4.png)

## Question 4

### A

`TestSeq` iterative optical flow:

![ps5-4-a-1.png](output/ps5-4-a-1.png)

`TestSeq` iterative difference:

![ps5-4-a-2.png](output/ps5-4-a-2.png)

### B

`DataSeq1` iterative optical flow:

![ps5-4-b-1.png](output/ps5-4-b-1.png)

`DataSeq1` iterative difference:

![ps5-4-b-2.png](output/ps5-4-b-2.png)

### C

`DataSeq2` iterative optical flow:

![ps5-4-c-1.png](output/ps5-4-c-1.png)

`DataSeq2` iterative difference:

![ps5-4-c-2.png](output/ps5-4-c-2.png)

## Question 5

### A

`Juggle` iterative optical flow:

![ps5-5-a-1.png](output/ps5-5-a-1.png)

`Juggle` iterative difference:

![ps5-5-a-2.png](output/ps5-5-a-2.png)