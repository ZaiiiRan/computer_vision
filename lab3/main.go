package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
)

func buildKernel(size int, sigma float64) [][]float64 {
	matr := make([][]float64, size)
	for i := range matr {
		matr[i] = make([]float64, size)
	}

	a := int(size-size/2) - 1
	b := a

	for x := 0; x < size; x++ {
		for y := 0; y < size; y++ {
			matr[x][y] = (1 / (2 * math.Pi * math.Pow(sigma, 2))) *
				math.Exp(-((math.Pow(float64(x)-float64(a), 2) + math.Pow(float64(y)-float64(b), 2)) / (2 * math.Pow(sigma, 2))))
		}
	}

	return matr
}

func normKernel(kernel [][]float64) [][]float64 {
	var sum float64
	for _, row := range kernel {
		for _, v := range row {
			sum += v
		}
	}

	for i := range kernel {
		for j := range kernel[i] {
			kernel[i][j] /= sum
		}
	}

	return kernel
}

func gaussianBlur(img image.Image, kernelSize int, blurParameter float64) *image.Gray {
	kernel := normKernel(buildKernel(kernelSize, blurParameter))

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	padding := kernelSize / 2

	blurred := image.NewGray(bounds)

	for y := range height {
		for x := range width {
			var sum float64

			for ky := range kernelSize {
				for kx := range kernelSize {
					px := x + kx - padding
					py := y + ky - padding

					if px < 0 {
						px = -px
					} else if px >= width {
						px = 2*width - px - 1
					}
					if py < 0 {
						py = -py
					} else if py >= height {
						py = 2*height - py - 1
					}

					r, g, b, _ := img.At(px, py).RGBA()
					intensity := float64(((r + g + b) / 3) / 256)
					sum += intensity * kernel[ky][kx]
				}
			}

			if sum < 0 {
				sum = 0
			} else if sum > 255 {
				sum = 255
			}

			blurred.SetGray(x, y, color.Gray{Y: uint8(sum)})
		}
	}

	return blurred
}

func main() {
	file, err := os.Open("../lab1/imgs/cat.jpg")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		panic(err)
	}

	kernelSize := 7
	blurParameter := 15.0

	blurred := gaussianBlur(img, kernelSize, blurParameter)

	outFile, _ := os.Create("blurred.jpg")
	defer outFile.Close()
	jpeg.Encode(outFile, blurred, nil)
}
