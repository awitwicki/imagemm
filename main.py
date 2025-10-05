import numpy as np
import tensorflow as tf
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
import os

class MonoImageMM:
    def __init__(self):
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Using device: {self.device}")
    
    def load_mono_frames(self, file_paths):
        """
        Завантаження монохромних .fit файлів
        """
        print("Loading mono frames...")
        frames = []
        for file_path in tqdm(file_paths, desc="Loading FITS files"):
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                frames.append(data)
        
        # Конвертуємо у TensorFlow tensor та розміщуємо на GPU
        with tf.device(self.device):
            frames_tensor = tf.constant(np.array(frames), dtype=tf.float32)
        return frames_tensor
    
    def estimate_psf_from_stars(self, frame, psf_size=25, star_threshold=0.1):
        """
        Оцінка PSF з зірок на монохромному зображенні
        """
        from skimage.feature import peak_local_max
        
        # Нормалізація зображення
        frame_normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8)
        
        # Виявлення яскравих зірок
        star_coordinates = peak_local_max(
            frame_normalized,
            min_distance=1,
            threshold_rel=star_threshold,
            num_peaks=15,
            exclude_border=20
        )
        
        print(f"Found {len(star_coordinates)} stars for PSF estimation")
        
        star_patches = []
        for y, x in star_coordinates:
            if (y >= psf_size//2 and y < frame.shape[0] - psf_size//2 and 
                x >= psf_size//2 and x < frame.shape[1] - psf_size//2):
                
                patch = frame[y-psf_size//2:y+psf_size//2+1, 
                             x-psf_size//2:x+psf_size//2+1]
                
                normalized_patch = patch / np.sum(patch)
                star_patches.append(normalized_patch)
        
        if len(star_patches) >= 3:
            median_psf = np.median(star_patches, axis=0)
            psf_result = median_psf / np.sum(median_psf)
        else:
            print("Not enough good stars found, using Gaussian PSF")
            psf_result = self.create_gaussian_psf(psf_size, sigma=2.0)
        
        # Конвертуємо PSF у TensorFlow tensor
        with tf.device(self.device):
            psf_tensor = tf.constant(psf_result, dtype=tf.float32)
        return psf_tensor
    
    def create_gaussian_psf(self, size, sigma=2.0):
        """Створення гаусівської PSF"""
        x = np.linspace(-size//2, size//2, size)
        y = np.linspace(-size//2, size//2, size)
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return gaussian / np.sum(gaussian)
    
    def simultaneous_multi_frame_processing(self, frames, psfs, max_iterations=50, convergence_threshold=1e-6):
        """
        Основний алгоритм ImageMM для монохромних зображень з використанням GPU
        """
        with tf.device(self.device):
            n_frames, height, width = frames.shape
            
            # 1. Ініціалізація - медіана кадрів
            print("Initializing with median frame...")
            latent_image = tf.math.reduce_mean(frames, axis=0)
            latent_image = tf.Variable(latent_image, dtype=tf.float32)
            
            # 2. Основний цикл MM
            previous_loss = float('inf')
            
            for iteration in tqdm(range(max_iterations), desc="MM Processing"):
                numerator = tf.zeros_like(latent_image)
                denominator = tf.zeros_like(latent_image)
                total_loss = 0.0
                
                for t in range(n_frames):
                    # Симуляція спостереження з використанням TensorFlow згортки
                    # Підготовка тензорів для згортки
                    latent_expanded = tf.reshape(latent_image, [1, height, width, 1])
                    psf_expanded = tf.reshape(psfs[t], [psfs[t].shape[0], psfs[t].shape[1], 1, 1])
                    
                    # Згортка з використанням tf.nn.conv2d
                    simulated_observation = tf.nn.conv2d(
                        latent_expanded, psf_expanded, 
                        strides=[1, 1, 1, 1], 
                        padding='SAME'
                    )
                    simulated_observation = tf.reshape(simulated_observation, [height, width])
                    
                    # Розрахунок втрат
                    frame_loss = tf.reduce_mean((frames[t] - simulated_observation) ** 2)
                    total_loss += frame_loss.numpy()
                    
                    # Коефіцієнт оновлення
                    observation_ratio = frames[t] / (simulated_observation + 1e-12)
                    
                    # Зворотне поширення з використанням транспонованої PSF
                    transposed_psf = tf.reverse(psfs[t], axis=[0, 1])  # Еквівалент ротації на 180 градусів
                    transposed_psf = tf.reshape(transposed_psf, [psfs[t].shape[0], psfs[t].shape[1], 1, 1])
                    observation_ratio_expanded = tf.reshape(observation_ratio, [1, height, width, 1])
                    
                    backward_correction = tf.nn.conv2d(
                        observation_ratio_expanded, transposed_psf,
                        strides=[1, 1, 1, 1],
                        padding='SAME'
                    )
                    backward_correction = tf.reshape(backward_correction, [height, width])
                    
                    numerator += backward_correction
                    denominator += 1.0
                
                # Оновлення зображення
                update_factor = numerator / (denominator + 1e-12)
                latent_image.assign(latent_image * update_factor)
                
                # Фізичні обмеження
                latent_image.assign(tf.maximum(latent_image, 0))
                
                # Перевірка збіжності
                current_loss = total_loss / n_frames
                loss_change = abs(previous_loss - current_loss) / previous_loss
                
                if iteration % 5 == 0:
                    print(f"Iteration {iteration}: Loss = {current_loss:.6f}, Change = {loss_change:.6f}")
                
                if loss_change < convergence_threshold and iteration > 10:
                    print(f"Converged after {iteration} iterations")
                    break
                    
                previous_loss = current_loss
            
            return latent_image.numpy()
    
    def enhance_final_result(self, image, contrast_percentile=99.5):
        """Покращення контрасту фінального зображення"""
        vmax = np.percentile(image, contrast_percentile)
        enhanced = np.minimum(image, vmax)
        enhanced = enhanced / np.max(enhanced)
        return enhanced
    
    def run_mono_workflow(self, file_paths):
        """
        Повний робочий процес для монохромних зображень
        """
        print("=== Mono ImageMM Workflow ===")
        
        # 1. Завантаження кадрів
        print("1. Loading mono frames...")
        frames = self.load_mono_frames(file_paths)
        print(f"Loaded {len(frames)} frames of size {frames[0].shape}")
        
        # 2. Оцінка PSF
        print("2. Estimating PSFs from stars...")
        psfs = []
        for i, frame in enumerate(tqdm(frames, desc="PSF Estimation")):
            frame_np = frame.numpy()  # Конвертуємо для PSF estimation
            psf = self.estimate_psf_from_stars(frame_np)
            psfs.append(psf)
        
        # 3. ImageMM обробка
        print("3. Running ImageMM restoration...")
        restored_image = self.simultaneous_multi_frame_processing(frames, psfs)
        
        # 4. Порівняння з традиційними методами
        print("4. Generating results...")
        single_frame = frames[0].numpy()
        median_stack = np.median(frames.numpy(), axis=0)
        
        # Покращення контрасту для відображення
        single_enhanced = self.enhance_final_result(single_frame)
        median_enhanced = self.enhance_final_result(median_stack)
        restored_enhanced = self.enhance_final_result(restored_image)
        
        # Візуалізація
        self.create_mono_comparison(single_enhanced, median_enhanced, restored_enhanced)
        
        return restored_image, median_stack, frames.numpy(), psfs

    def create_mono_comparison(self, single_frame, median_stack, imageMM_result):
        """Створення порівняльних зображень для монохромних даних"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        cmap = 'gray'
        
        # Оригінальний кадр
        axes[0, 0].imshow(single_frame, cmap=cmap, vmin=0, vmax=1)
        axes[0, 0].set_title('Single Mono Frame\n(Noisy & Blurry)')
        axes[0, 0].axis('off')
        
        # Медіанний стак
        axes[0, 1].imshow(median_stack, cmap=cmap, vmin=0, vmax=1)
        axes[0, 1].set_title('Median Stack\n(Reduced Noise)')
        axes[0, 1].axis('off')
        
        # ImageMM результат
        axes[0, 2].imshow(imageMM_result, cmap=cmap, vmin=0, vmax=1)
        axes[0, 2].set_title('ImageMM Restoration\n(Sharp & Clean)')
        axes[0, 2].axis('off')
        
        # Деталі - обрізка центральної частини
        crop_size = min(200, imageMM_result.shape[0]//4, imageMM_result.shape[1]//4)
        h, w = imageMM_result.shape[0]//2, imageMM_result.shape[1]//2
        crop_start_h = h - crop_size//2
        crop_start_w = w - crop_size//2
        
        single_crop = single_frame[crop_start_h:crop_start_h+crop_size, 
                                  crop_start_w:crop_start_w+crop_size]
        median_crop = median_stack[crop_start_h:crop_start_h+crop_size, 
                                  crop_start_w:crop_start_w+crop_size]
        imageMM_crop = imageMM_result[crop_start_h:crop_start_h+crop_size, 
                                     crop_start_w:crop_start_w+crop_size]
        
        # Порівняння деталей
        axes[1, 0].imshow(single_crop, cmap=cmap, vmin=0, vmax=1)
        axes[1, 0].set_title('Single Frame Detail')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(median_crop, cmap=cmap, vmin=0, vmax=1)
        axes[1, 1].set_title('Median Stack Detail')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(imageMM_crop, cmap=cmap, vmin=0, vmax=1)
        axes[1, 2].set_title('ImageMM Detail')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('output/mono_imageMM_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Профілі яскравості
        self.plot_intensity_profiles(single_frame, median_stack, imageMM_result)

    def plot_intensity_profiles(self, single, median, restored):
        """Побудова профілів яскравості"""
        center_h, center_w = restored.shape[0]//2, restored.shape[1]//2
        crop_size = min(200, restored.shape[0]//4, restored.shape[1]//4)
        
        center_region = restored[center_h-crop_size//2:center_h+crop_size//2, 
                                center_w-crop_size//2:center_w+crop_size//2]
        max_pos = np.unravel_index(np.argmax(center_region), center_region.shape)
        star_y = center_h - crop_size//2 + max_pos[0]
        star_x = center_w - crop_size//2 + max_pos[1]
        
        # Беремо горизонтальний зріз через зірку
        profile_single = single[star_y, star_x-50:star_x+50]
        profile_median = median[star_y, star_x-50:star_x+50]
        profile_restored = restored[star_y, star_x-50:star_x+50]
        
        plt.figure(figsize=(12, 6))
        plt.plot(profile_single, 'b-', label='Single Frame', alpha=0.7)
        plt.plot(profile_median, 'g-', label='Median Stack', alpha=0.7)
        plt.plot(profile_restored, 'r-', label='ImageMM', linewidth=2)
        plt.title('Intensity Profiles Through a Star')
        plt.xlabel('Pixel Position')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('output/mono_intensity_profiles.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    processor = MonoImageMM()
    
    # Автоматичний пошук .fit файлів у поточній папці
    print("Searching for .fit files in current directory...")
    fit_files = ['input/' + f for f in os.listdir('input') if f.lower().endswith('.fit')]
    if fit_files:
        file_paths = fit_files
        print(f"Found {len(fit_files)} .fit files, using first {len(file_paths)}")
    else:
        print("No .fit files found in current directory!")
        return
    
    print(f"Processing {len(file_paths)} files:")
    for f in file_paths:
        print(f"  - {f}")
    
    # Запуск робочого процесу
    try:
        restored_image, median_stack, original_frames, psfs = processor.run_mono_workflow(file_paths)
        
        # Збереження результатів
        from astropy.io import fits
        
        hdu_original = fits.PrimaryHDU(restored_image.astype(np.float32))
        hdu_original.writeto('output/imageMM_mono_restored.fits', overwrite=True)
        
        hdu_median = fits.PrimaryHDU(median_stack.astype(np.float32))
        hdu_median.writeto('output/median_mono_stack.fits', overwrite=True)
        
        print("Processing complete!")
        print(f"Processed {len(original_frames)} frames")
        print(f"Results saved:")
        print(f"- imageMM_mono_restored.fits")
        print(f"- median_mono_stack.fits") 
        print(f"- mono_imageMM_comparison.png")
        print(f"- mono_intensity_profiles.png")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()