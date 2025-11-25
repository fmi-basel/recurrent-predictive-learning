import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import kornia as K
import glob
from collections import defaultdict
from matplotlib import cm
from repl.data.utils import seed_worker

# Add multiprocessing configuration at the beginning
import torch.multiprocessing as mp


class SpriteVideoDataset(Dataset):
    """
    Dataset for creating synthetic videos from sprite images with trajectory generation.
    
    Args:
        data_dir (str): Directory containing the sprite images
        split (str): Dataset split to use ('train' or 'test')
        output_size (Tuple[int, int]): Width and height of output frames
        seq_len (int): Length of each video sequence
        num_sequences (int): Number of sequences to generate
        background (float): Background color intensity (0.0 to 1.0)
        device (str): Device to perform transformations on ('cuda' or 'cpu')
        seed (int): Random seed for reproducibility
        max_sprites (int): Maximum number of sprites to use (use first n only)
        exclude_latent_regions (bool): Whether to exclude latent regions (enforced that this only happens during training)
        sprite_img_dir (str): Directory containing sprite images
        discretize_latents (bool): Whether to discretize the latent space for continuous variables
        
        # Background noise options
        noise_type (str): Type of background noise: 'salt_pepper', 'gaussian'
        noise_intensity (float): Intensity of the noise (0.0 to 1.0)
        freeze_noise (bool): Whether to use the same noise pattern for all frames
        
        # Background grid options
        grid_enabled (bool): Whether to add a grid of parallel lines to the background
        freeze_grid (bool): Whether to use the same grid pattern for all frames

        # Occlusion option
        occlude_n_frames (int): Number of frames to occlude (0 means no occlusion)

        # Normalization parameters
        mean (float): Pre-computed mean for normalization (for test split)
        std (float): Pre-computed std for normalization (for test split)

        download (bool): Whether to download the dataset (not used here, included for API consistency)  
        transform (callable): Optional transform to be applied on a sample (not used here, included for API consistency)
    """
    
    valid_splits = ["train", "test"]
    valid_noise_types = ["salt_pepper", "gaussian"]
    
    def __init__(
        self,
        data_dir,
        split="train",
        output_size=(224, 224),
        seq_len=32,
        num_sequences=1000,
        background=0.5,
        grayscale=False,
        # device='cpu',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        max_sprites=None,  # parameter for limiting sprite count
        sprite_img_dir="animals", # parameter for sprite image directory
        exclude_latent_regions=False,  # parameter for excluding latent regions (usually during training)
        discretize_latents=False,  # parameter for discretizing latent space
        download=False,  # Included for API consistency, not used
        transform=None,  # Included for API consistency, not used
        
        # Background noise options
        noise_type=None,
        noise_intensity=0.1,
        freeze_noise=True,
        noise_on_top=False,  # Whether to add noise on top of the frame or as background
        
        # Background grid options
        grid_enabled=False,
        freeze_grid=True,

        # Occlusion option
        occlude_n_frames=0,
        
        # Normalization parameters
        mean=None,  # Pre-computed mean (for test split)
        std=None,   # Pre-computed std (for test split)
    ):
        assert split in self.valid_splits, f"split must be one of {self.valid_splits}"
        if noise_type:
            assert noise_type in self.valid_noise_types, f"noise_type must be one of {self.valid_noise_types}"
        
        self.data_dir = data_dir
        self.split = split
        self.output_size = output_size
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.background = background
        self.grayscale = grayscale
        self.device = device
        self.seed = seed
        self.max_sprites = max_sprites
        self.exclude_latent_regions = exclude_latent_regions
        self.sprite_img_dir = sprite_img_dir
        self.discretize_latents = discretize_latents
        
        # noise options
        self.noise_type = noise_type
        self.noise_intensity = noise_intensity
        self.freeze_noise = freeze_noise

        self.noise_on_top = noise_on_top  # Whether to add noise on top of the frame or as background
        
        # Background grid options
        self.grid_enabled = grid_enabled
        self.freeze_grid = freeze_grid

        self.occlude_n_frames = occlude_n_frames
        
        # Set random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        
        # Define discretization levels for each continuous variable
        # These are derived from the ranges used in __getitem__
        self.discretization_config = {
            'xpos': {'levels': 33, 'min': -1.0, 'max': 1.0},  # 33 levels (0-32) mapping [-1, 1]
            'ypos': {'levels': 33, 'min': -1.0, 'max': 1.0},  # 33 levels (0-32) mapping [-1, 1]
            'zpos': {'levels': 17, 'min': 0.0, 'max': 1.0},   # 17 levels (0-16) mapping [0, 1]
            'theta': {'levels': 36, 'min': 0.0, 'max': 1.0},  # 36 levels (0-35) mapping [0, 1] rotation
            'xvel': {'levels': 9, 'min': -1.0, 'max': 1.0},  # 9 levels mapping [-1, 1]
            'yvel': {'levels': 9, 'min': -1.0, 'max': 1.0},  # 9 levels mapping [-1, 1]
            'zvel': {'levels': 5, 'min': -1.0, 'max': 1.0},  # 5 levels mapping [-1, 1]
            'angular_speed': {'levels': 9, 'min': 0.0, 'max': 1.0}  # 9 levels mapping [0, 1]
        }
        
        # Load sprites (limited to max_sprites if specified)
        self.sprites = self._load_sprites()
        
        # Initialize base background (this will be modified with noise/grid)
        self.base_background_tensor = self._create_base_background(background).to(self.device)
        
        # Generate trajectories for train set
        if split == "train":
            self.train = True
            self.trajectories, self.labels = self._generate_trajectories()
            self.indices = list(range(self.num_sequences))  # Use all sequences for training
        else:  # load trajectories for test set
            self.train = False
            self.trajectories, self.labels = self._load_trajectories()
            self.indices = list(range(len(self.trajectories)))  # Use all sequences for testing
        
        # Split data
        # train_size = int(0.8 * self.num_sequences)
        # if split == "train":
        #     self.indices = list(range(train_size))
        #     self.train = True
        # else:  # test
        #     self.indices = list(range(train_size, self.num_sequences))
        #     self.train = False
        
        # Normalization
        if split == "train":
            self.mean, self.std = self._compute_normalization(
                samples=10000, 
                batch_size=100
            )
            if self.grayscale:
                print(f"Computed normalization values - mean: {self.mean.item():.4f}, std: {self.std.item():.4f}")
            else:
                print(f"Computed normalization values - mean: {self.mean}, std: {self.std}")
        else:
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for test split")
            self.mean = mean
            self.std = std

    def _compute_normalization(self, samples=10000, batch_size=100):
        """
        Compute mean and standard deviation across a subset of the dataset
        for normalization purposes.
        
        Args:
            samples (int): Number of samples to use for computation
            batch_size (int): Batch size to use for computation to save memory
            
        Returns:
            tuple: (mean, std) Computed mean and standard deviation
        """
        import torch
        print(f"Computing normalization values using {samples} samples...")
        
        # Determine how many samples to actually use
        samples = min(samples, len(self.indices))
        
        # Number of batches
        num_batches = (samples + batch_size - 1) // batch_size

        n_channels = 1 if self.grayscale else 3
        
        # Use running statistics to avoid loading all videos into memory
        mean = torch.zeros(n_channels, device=self.device)
        std = torch.zeros(n_channels, device=self.device)
        
        # Temporarily store original indices
        original_indices = self.indices.copy()
        
        try:
            # Use a subset of indices for computation
            self.indices = self.indices[:samples]
            
            # Process in batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, samples)
                
                # Generate each video in the batch
                for i in range(start_idx, end_idx):
                    # Map to actual index in trajectories
                    traj_idx = self.indices[i]
                    
                    # Get the sprite index and transformation sequence
                    sprite_idx, transform_sequence = self.trajectories[traj_idx]
                    
                    # Create the video (without normalization)
                    video = self._create_video(sprite_idx, transform_sequence)
                    
                    if self.grayscale:
                        mean += video.mean().item() / samples
                        std += video.std().item() / samples
                    else:
                        mean += video.mean(dim=[0, 2, 3]) / samples
                        std += video.std(dim=[0, 2, 3]) / samples
        
        finally:
            # Restore original indices
            self.indices = original_indices
        
        if self.grayscale:
            std = max(1e-5, std)  # Ensure std is not too small
        else:
            std = torch.clamp(std, min=1e-5)  # Ensure std is not too small

        # Compute mean and standard deviation
        return mean, std

    def _load_trajectories(self):
        """
        Load pre-generated trajectories from disk.
        
        Returns:
            trajectories (list): List of transformation sequences
            labels (dict): Dictionary of trajectory labels
        """
        import pickle
        
        trajectories = []
        labels = defaultdict(list)

        cts = "dscrt" if self.discretize_latents else "cts"

        traj_file = os.path.join(self.data_dir, f"sprite_{cts}_latents_{self.split}_trajectories.pkl")
        labels_file = os.path.join(self.data_dir, f"sprite_{cts}_latents_{self.split}_labels.pkl")

        if not os.path.exists(traj_file) or not os.path.exists(labels_file):
            raise FileNotFoundError(f"Trajectories or labels file not found: {traj_file} or {labels_file}")

        with open(traj_file, 'rb') as f:
            trajectories = pickle.load(f)

        with open(labels_file, 'rb') as f:
            labels = pickle.load(f)

        return trajectories, labels

    def _load_sprites(self):
        """Load all sprite images from the data directory as grayscale with alpha."""
        sprite_paths = sorted(glob.glob(os.path.join(self.data_dir, self.sprite_img_dir, "sprite_*.png")))
        assert len(sprite_paths) > 0, f"No sprites found in {self.data_dir}/{self.sprite_img_dir}"
        
        # Limit sprites if max_sprites is specified
        if self.max_sprites is not None:
            sprite_paths = sprite_paths[:self.max_sprites]
        
        sprites = []
        for path in sprite_paths:
            # Load as RGBA to preserve alpha channel
            img = Image.open(path).convert('RGBA')
            # Convert to grayscale+alpha
            img_tensor = self._pil_to_tensor(img).to(self.device)
            sprites.append(img_tensor)
        
        return sprites
    
    def _pil_to_tensor(self, pil_img):
        if pil_img.mode == 'RGBA':
            """Convert PIL RGBA image to tensor based on grayscale flag"""
            # For RGBA images, handle based on grayscale flag
            rgb = Image.new('RGB', pil_img.size, (0, 0, 0))
            rgb.paste(pil_img, mask=pil_img.split()[3])  # Use alpha as mask
            
            # Get alpha channel as separate tensor
            alpha_np = np.array(pil_img.split()[3]).astype(np.float32) / 255.0
            
            if self.grayscale:
                # Convert RGB to grayscale
                gray = rgb.convert('L')
                # Create grayscale tensor
                gray_np = np.array(gray).astype(np.float32) / 255.0
                # Stack grayscale and alpha
                img_np = np.stack([gray_np, alpha_np], axis=0)  # [2, H, W]
            else:
                # Keep RGB channels
                rgb_np = np.array(rgb).astype(np.float32) / 255.0
                # Convert to channels-first and stack with alpha
                rgb_np = np.transpose(rgb_np, (2, 0, 1))  # [3, H, W]
                img_np = np.concatenate([rgb_np, alpha_np[None]], axis=0)  # [4, H, W]
                
            img_tensor = torch.from_numpy(img_np)
        else:
            # For RGB/grayscale images
            gray = pil_img.convert('L')
            img_np = np.array(gray).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]
        
        return img_tensor
    
    def _create_base_background(self, background=0.5):
        """Create a base grayscale background tensor"""
        h, w = self.output_size
        n_channels = 1 if self.grayscale else 3
        
        # Default gray background
        return torch.full((n_channels, h, w), background, device=self.device)

        # # Default black background
        # return torch.zeros(n_channels, h, w, device=self.device)

    def _generate_noise_for_video(self, seq_len):
        """
        Generate noise for all frames in a video at once
        
        Args:
            seq_len (int): Number of frames in the video
            
        Returns:
            torch.Tensor: Noise for all frames, shape [seq_len, 1, H, W]
        """
        h, w = self.output_size
        n_channels = 1 if self.grayscale else 3
        
        if self.noise_type is None or self.noise_intensity == 0:
            return torch.zeros(seq_len, n_channels, h, w, device=self.device)
        
        if self.freeze_noise:
            # Generate one noise pattern and repeat it for all frames
            if self.noise_type == "salt_pepper":
                # Create salt and pepper noise (binary noise)
                noise = torch.rand(1, n_channels, h, w, device=self.device)
                noise = (noise < self.noise_intensity/2).float() - (noise > (1 - self.noise_intensity/2)).float()
            elif self.noise_type == "gaussian":
                # Create Gaussian noise
                noise = torch.randn(1, n_channels, h, w, device=self.device) * self.noise_intensity
            
            # Repeat the same noise for all frames
            return noise.repeat(seq_len, 1, 1, 1)
        else:
            # Generate different noise for each frame
            if self.noise_type == "salt_pepper":
                # Create salt and pepper noise (binary noise)
                noise = torch.rand(seq_len, n_channels, h, w, device=self.device)
                noise = (noise < self.noise_intensity/2).float() - (noise > (1 - self.noise_intensity/2)).float()
            elif self.noise_type == "gaussian":
                # Create Gaussian noise
                noise = torch.randn(seq_len, n_channels, h, w, device=self.device) * self.noise_intensity
            
            return noise

    def _generate_grid_for_video(self, seq_len):
        """
        Generate grid patterns for all frames in a video at once
        
        Args:
            seq_len (int): Number of frames in the video
            
        Returns:
            torch.Tensor: Grid for all frames, shape [seq_len, 1, H, W]
        """
        h, w = self.output_size
        n_channels = 1 if self.grayscale else 3
        
        if not self.grid_enabled:
            return torch.zeros(seq_len, n_channels, h, w, device=self.device)
        
        # Create coordinate meshgrids once for all frames
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=self.device),
            torch.arange(w, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        # Center the coordinates (helps with rotation)
        x_centered = x_coords - w/2
        y_centered = y_coords - h/2
        
        # Sample grid parameters once per video (regardless of freeze_grid)
        grid_spacing = self.rng.randint(10, 101)  # Between 10 and 100
        grid_angle = self.rng.uniform(0, 360)  # Any angle
        grid_color = self.rng.uniform(0.5, 1.0)  # Between 0.5 and 1.0
        if not self.grayscale:
            grid_color = torch.tensor([grid_color, grid_color, grid_color], device=self.device)
        grid_speed = self.rng.uniform(1, 16)  # Between 1 and 16
        grid_thickness = max(1, int(grid_spacing * 0.05))  # Scale thickness with spacing
        
        # Convert grid angle to radians
        angle_rad = torch.tensor(grid_angle * (torch.pi / 180.0), device=self.device)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # Rotate coordinates
        x_rot = x_centered * cos_a - y_centered * sin_a
        y_rot = x_centered * sin_a + y_centered * cos_a
        
        # Create frame offsets for all frames at once
        frame_indices = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        
        # Apply movement only if freeze_grid is False
        if not self.freeze_grid:
            offsets = (frame_indices * grid_speed).unsqueeze(-1).unsqueeze(-1) % grid_spacing
            
            # Apply offsets to all frames (broadcasting)
            x_rot_with_offset = x_rot.unsqueeze(0) + offsets
        else:
            # No movement for frozen grid
            x_rot_with_offset = x_rot.unsqueeze(0).expand(seq_len, -1, -1)
        
        # Apply modulo to get repeating pattern across all frames
        x_mod = torch.fmod(torch.abs(x_rot_with_offset), grid_spacing)
        y_mod = torch.fmod(torch.abs(y_rot.unsqueeze(0)), grid_spacing)
        
        # Create grid pattern - both horizontal and vertical lines for all frames
        grid_pattern_h = x_mod < grid_thickness
        grid_pattern_v = y_mod < grid_thickness
        grid_pattern = (grid_pattern_h | grid_pattern_v).unsqueeze(1)  # Combine both
        
        # Set grid lines to specified color
        grids = torch.zeros(seq_len, n_channels, h, w, device=self.device)
        grids[grid_pattern] = grid_color
        
        return grids

    def _create_backgrounds_for_video(self, seq_len):
        """
        Create backgrounds for all frames of a video at once in a fully vectorized way
        
        Args:
            seq_len (int): Number of frames in the video
            
        Returns:
            torch.Tensor: Backgrounds for all frames, shape [seq_len, 1, H, W]
        """
        h, w = self.output_size
        n_channels = 1 if self.grayscale else 3
        
        # Start with base background repeated for all frames
        backgrounds = self.base_background_tensor.unsqueeze(0).repeat(seq_len, 1, 1, 1)
        
        # Add noise if enabled and is background only (not on top)
        if not self.noise_on_top:
            if self.noise_type:
                noise = self._generate_noise_for_video(seq_len)
                backgrounds = torch.clamp(backgrounds + noise, 0.0, 1.0)
        
        # Add grid if enabled
        if self.grid_enabled:
            grids = self._generate_grid_for_video(seq_len)
            grid_mask = (grids > 0).float()
            backgrounds = backgrounds * (1 - grid_mask) + grids * grid_mask
        
        return backgrounds
    
    def _generate_continuous_trajectories(self):
        """
        Generate trajectories for sprites using continuous dynamics.
        
        Returns:
            trajectories (list): List of transformation sequences
            labels (dict): Dictionary of trajectory labels
        """
        trajectories = []
        
        # Create dictionary for all labels
        labels = {
            # Categorical labels
            'sprite_idx': np.zeros(self.num_sequences, dtype=np.int32),  # Index of sprite used
            'rotation_dir': np.zeros(self.num_sequences),          # 0=constant, 1=ccw, 2=cw

            # Continuous values
            'speed': np.zeros(self.num_sequences),                 # Overall speed (magnitude)
            'angular_speed': np.zeros(self.num_sequences),         # Angular speed (magnitude)

            # Per-timestep categorical labels
            'xdir': np.zeros((self.num_sequences, self.seq_len)),  # 0=constant, 1=right, 2=left
            'ydir': np.zeros((self.num_sequences, self.seq_len)),  # 0=constant, 1=up, 2=down
            'zdir': np.zeros((self.num_sequences, self.seq_len)),  # 0=constant, 1=increase, 2=decrease
            
            # Per-timestep continuous labels
            'xpos': np.zeros((self.num_sequences, self.seq_len)),  # X position
            'ypos': np.zeros((self.num_sequences, self.seq_len)),  # Y position
            'zpos': np.zeros((self.num_sequences, self.seq_len)),  # Z position (scale)
            'theta': np.zeros((self.num_sequences, self.seq_len)), # Rotation angle
            'xvel': np.zeros((self.num_sequences, self.seq_len)),  # Velocity along x-axis
            'yvel': np.zeros((self.num_sequences, self.seq_len)),  # Velocity along y-axis
            'zvel': np.zeros((self.num_sequences, self.seq_len)),  # Velocity along z-axis (scale change)
        }
        
        w, h = self.output_size
        max_vel = 8  # Maximum velocity in pixels per frame
        max_zvel = 0.125  # Maximum z velocity (scale change per frame)
        
        for i in range(self.num_sequences):
            # Randomly select a sprite
            sprite_idx = self.rng.randint(0, len(self.sprites))
            labels['sprite_idx'][i] = sprite_idx
            
            # Uniform sampling of velocities in all directions
            # Sample x velocity directly from [-max_vel, max_vel]
            xvel_initial = self.rng.uniform(-max_vel, max_vel)
            # Sample y velocity directly from [-max_vel, max_vel]
            yvel_initial = self.rng.uniform(-max_vel, max_vel)
            # Sample z velocity directly from [-max_zvel, max_zvel]
            zvel_initial = self.rng.uniform(-max_zvel, max_zvel)
            
            # Uniform sampling for rotation velocity
            rotation_speed = self.rng.uniform(-30, 30)  # Degrees per frame
            
            # Leave out chunks of latent regions during training
            # Only possible for xvel, yvel, zvel and rotation_speed.
            # Everything else is determined by the physics of the simulation.
            if self.train and self.exclude_latent_regions:
                Warning("Excluding latent regions")
                # Let's leave out x velocities between ±0.25*max_vel and ±0.75*max_vel
                # Similarly, let's leave out y velocities between -0.25*max_vel and 0.25*max_vel
                # With z velocities, let's leave out -0.25*max_zvel to 0.25*max_zvel
                # Let's not subsample rotation speeds for now
                
                # Domain ranges for x velocities: [-max_vel, -0.75*max_vel] ∪ [-0.25*max_vel, 0.25*max_vel] ∪ [0.75*max_vel, max_vel]
                # For y and z: [-max_vel, -0.25*max_vel] ∪ [0.25*max_vel, max_vel]
                # Sample x velocity from restricted domain
                x_region = self.rng.choice([0, 1, 2])  # 0: left high, 1: middle, 2: right high
                if x_region == 0:
                    xvel_initial = self.rng.uniform(-max_vel, -0.75 * max_vel)
                elif x_region == 1:
                    xvel_initial = self.rng.uniform(-0.25 * max_vel, 0.25 * max_vel)
                else:  # x_region == 2
                    xvel_initial = self.rng.uniform(0.75 * max_vel, max_vel)
                
                # Sample y velocity from restricted domain (excluding low velocities)
                y_region = self.rng.choice([0, 1])  # 0: negative high, 1: positive high
                if y_region == 0:
                    yvel_initial = self.rng.uniform(-max_vel, -0.25 * max_vel)
                else:  # y_region == 1
                    yvel_initial = self.rng.uniform(0.25 * max_vel, max_vel)
                
                # Sample z velocity from restricted domain (excluding low velocities)
                z_region = self.rng.choice([0, 1])  # 0: negative high, 1: positive high
                if z_region == 0:
                    zvel_initial = self.rng.uniform(-max_zvel, -0.25 * max_zvel)
                else:  # z_region == 1
                    zvel_initial = self.rng.uniform(0.25 * max_zvel, max_zvel)
            
            # Set initial direction labels based on sampled velocities
            if xvel_initial > 0:
                labels['xdir'][i, 0] = 1  # Moving right
            elif xvel_initial < 0:
                labels['xdir'][i, 0] = 2  # Moving left
            else:
                labels['xdir'][i, 0] = 0  # Not moving in x
                
            if yvel_initial > 0:
                labels['ydir'][i, 0] = 2  # Moving down
            elif yvel_initial < 0:
                labels['ydir'][i, 0] = 1  # Moving up
            else:
                labels['ydir'][i, 0] = 0  # Not moving in y
                
            if zvel_initial > 0:
                labels['zdir'][i, 0] = 1  # Increasing
            elif zvel_initial < 0:
                labels['zdir'][i, 0] = 2  # Decreasing
            else:
                labels['zdir'][i, 0] = 0  # Not changing scale
            
            # Scale boundaries
            min_scale = 0.2  # Ensures at least 4x4 pixels for a 16x16 sprite
            max_scale = 1.0  # Sprite is 16x16 pixels
            
            # Sample initial positions (ensuring sprite is fully visible)
            padding = 8  # Ensure sprite isn't starting at the very edge
            posX = self.rng.randint(padding, w - padding)
            posY = self.rng.randint(padding, h - padding)
            
            # Sample initial scale uniformly across the entire range
            initial_scale = self.rng.uniform(min_scale, max_scale)
            
            # Set rotation direction label based on rotation speed
            if rotation_speed > 0:
                labels['rotation_dir'][i] = 1  # CCW
            elif rotation_speed < 0:
                labels['rotation_dir'][i] = 2  # CW
            else:
                labels['rotation_dir'][i] = 0  # Constant
            
            # Sample initial rotation uniformly
            initial_rotation = self.rng.uniform(0, 360)

            # Current velocities
            xvel_current = xvel_initial
            yvel_current = yvel_initial
            zvel_current = zvel_initial
            
            # Calculate and store speed (magnitude of velocity)
            labels['speed'][i] = np.sqrt(xvel_initial**2 + yvel_initial**2)
            
            # Store angular speed (magnitude)
            labels['angular_speed'][i] = abs(rotation_speed)
            
            # Generate sequence of transformations
            sequence = []
            
            # Generate each frame's transformation
            for t in range(self.seq_len):
                # Calculate initial position for t=0
                if t == 0:
                    current_posX = posX
                    current_posY = posY
                    current_scale = initial_scale
                else:
                    # Calculate next positions based on previous position and current velocity
                    current_posX = sequence[-1]['position'][0] + xvel_current
                    current_posY = sequence[-1]['position'][1] + yvel_current
                    current_scale = sequence[-1]['scale'] + zvel_current
                    
                    # Change padding based on scale
                    padding = int(8 * (current_scale / max_scale))

                    # Check for X boundary collisions and update velocity
                    if current_posX >= w - padding:
                        # Hitting right wall, reverse direction
                        xvel_current = -abs(xvel_current)
                        # Calculate reflection
                        current_posX = 2 * (w - padding) - current_posX
                        labels['xdir'][i, t] = 2  # Now moving left
                    elif current_posX < padding:
                        # Hitting left wall, reverse direction
                        xvel_current = abs(xvel_current)
                        # Calculate reflection
                        current_posX = 2 * padding - current_posX
                        labels['xdir'][i, t] = 1  # Now moving right
                    else:
                        # No collision
                        if xvel_current > 0:
                            labels['xdir'][i, t] = 1  # Moving right
                        elif xvel_current < 0:
                            labels['xdir'][i, t] = 2  # Moving left
                        else:
                            labels['xdir'][i, t] = 0  # Not moving in x
                    
                    # Check for Y boundary collisions and update velocity
                    if current_posY >= h - padding:
                        # Hitting bottom wall, reverse direction
                        yvel_current = -abs(yvel_current)
                        # Calculate reflection
                        current_posY = 2 * (h - padding) - current_posY
                        labels['ydir'][i, t] = 1  # Now moving up (negative y)
                    elif current_posY < padding:
                        # Hitting top wall, reverse direction
                        yvel_current = abs(yvel_current)
                        # Calculate reflection
                        current_posY = 2 * padding - current_posY
                        labels['ydir'][i, t] = 2  # Now moving down (positive y)
                    else:
                        # No collision
                        if yvel_current > 0:
                            labels['ydir'][i, t] = 2  # Moving down
                        elif yvel_current < 0:
                            labels['ydir'][i, t] = 1  # Moving up
                        else:
                            labels['ydir'][i, t] = 0  # Not moving in y
                    
                    # Check for Z boundary collisions and update velocity
                    if current_scale >= max_scale:
                        # Hitting max scale, reverse direction
                        zvel_current = -abs(zvel_current)
                        # Calculate reflection
                        current_scale = 2 * max_scale - current_scale
                        labels['zdir'][i, t] = 2  # Now decreasing
                    elif current_scale <= min_scale:
                        # Hitting min scale, reverse direction
                        zvel_current = abs(zvel_current)
                        # Calculate reflection
                        current_scale = 2 * min_scale - current_scale
                        labels['zdir'][i, t] = 1  # Now increasing
                    else:
                        # No collision
                        if zvel_current > 0:
                            labels['zdir'][i, t] = 1  # Increasing
                        elif zvel_current < 0:
                            labels['zdir'][i, t] = 2  # Decreasing
                        else:
                            labels['zdir'][i, t] = 0  # Not changing scale
                
                # Calculate rotation for this frame
                rotation = initial_rotation + rotation_speed * t
                
                # Store current transformation
                transform = {
                    'position': (current_posX, current_posY),
                    'rotation': rotation,
                    'scale': current_scale
                }
                sequence.append(transform)
                
                # Store current positions and velocities
                labels['xpos'][i, t] = (current_posX - w / 2) / (w / 2)  # Normalize to [-1, 1]
                labels['ypos'][i, t] = (current_posY - h / 2) / (h / 2)  # Normalize to [-1, 1]
                labels['zpos'][i, t] = (current_scale - min_scale) / (max_scale - min_scale)  # Normalize to [0, 1]
                labels['theta'][i, t] = (rotation % 360) / 360.0  # Normalize to [0, 1]
                labels['xvel'][i, t] = xvel_current / max_vel  # Normalize to [-1, 1]
                labels['yvel'][i, t] = yvel_current / max_vel  # Normalize to [-1, 1]
                labels['zvel'][i, t] = zvel_current / max_zvel  # Normalize to [-1, 1]
            
            trajectories.append((sprite_idx, sequence))
        
        return trajectories, labels

    def _generate_discrete_trajectories(self):
        """
        Generate trajectories for sprites using fully discrete dynamics.
        Dynamics are performed in discrete space and then converted back to continuous ranges.
        
        Returns:
            trajectories (list): List of transformation sequences
            labels (dict): Dictionary of trajectory labels
        """
        trajectories = []
        
        # Create dictionary for all labels
        labels = {
            # Categorical labels
            'sprite_idx': np.zeros(self.num_sequences, dtype=np.int32),  # Index of sprite used
            'rotation_dir': np.zeros(self.num_sequences),          # 0=constant, 1=ccw, 2=cw

            # Continuous values
            'speed': np.zeros(self.num_sequences),                 # Overall speed (magnitude)
            'angular_speed': np.zeros(self.num_sequences),         # Angular speed (magnitude)

            # Per-timestep categorical labels
            'xdir': np.zeros((self.num_sequences, self.seq_len)),  # 0=constant, 1=right, 2=left
            'ydir': np.zeros((self.num_sequences, self.seq_len)),  # 0=constant, 1=up, 2=down
            'zdir': np.zeros((self.num_sequences, self.seq_len)),  # 0=constant, 1=increase, 2=decrease
            
            # Per-timestep continuous labels
            'xpos': np.zeros((self.num_sequences, self.seq_len)),  # X position
            'ypos': np.zeros((self.num_sequences, self.seq_len)),  # Y position
            'zpos': np.zeros((self.num_sequences, self.seq_len)),  # Z position (scale)
            'theta': np.zeros((self.num_sequences, self.seq_len)), # Rotation angle
            'xvel': np.zeros((self.num_sequences, self.seq_len)),  # Velocity along x-axis
            'yvel': np.zeros((self.num_sequences, self.seq_len)),  # Velocity along y-axis
            'zvel': np.zeros((self.num_sequences, self.seq_len)),  # Velocity along z-axis (scale change)
        }
        
        w, h = self.output_size
        max_vel = 8  # Maximum velocity in pixels per frame
        max_zvel = 0.05  # Maximum z velocity (scale change per frame)
        
        # Scale boundaries
        min_scale = 0.2  # Ensures at least 4x4 pixels for a 16x16 sprite
        max_scale = 1.0  # Sprite is 16x16 pixels
        
        # Extract number of discrete levels for each dimension
        xpos_levels = self.discretization_config['xpos']['levels']
        ypos_levels = self.discretization_config['ypos']['levels']
        zpos_levels = self.discretization_config['zpos']['levels']
        theta_levels = self.discretization_config['theta']['levels']
        
        xvel_levels = self.discretization_config['xvel']['levels']
        yvel_levels = self.discretization_config['yvel']['levels']
        zvel_levels = self.discretization_config['zvel']['levels']
        angular_speed_levels = self.discretization_config['angular_speed']['levels']
        
        # Pre-calculate step sizes for each dimension
        xpos_step = 2.0 / (xpos_levels - 1)  # Range [-1, 1]
        ypos_step = 2.0 / (ypos_levels - 1)  # Range [-1, 1]
        zpos_step = 1.0 / (zpos_levels - 1)  # Range [0, 1]
        theta_step = 1.0 / (theta_levels - 1)  # Range [0, 1]
        
        # Step sizes for velocities should match position step sizes for consistent dynamics
        # For example, an x-velocity of 1 step should move position by 1 level
        xvel_step = xpos_step  # This ensures discrete velocity steps align with position steps
        yvel_step = ypos_step  # This ensures discrete velocity steps align with position steps
        zvel_step = zpos_step  # This ensures discrete velocity steps align with position steps
        
        # Calculate appropriate velocity ranges in discrete space
        max_discrete_xvel = (xvel_levels - 1) // 2  # Center is 0, range is symmetric
        max_discrete_yvel = (yvel_levels - 1) // 2
        max_discrete_zvel = (zvel_levels - 1) // 2
        
        # Calculate maximum angular speed in discrete space
        max_discrete_angular = (angular_speed_levels - 1)
        
        for i in range(self.num_sequences):
            # Randomly select a sprite
            sprite_idx = self.rng.randint(0, len(self.sprites))
            labels['sprite_idx'][i] = sprite_idx
            
            # Sample initial velocities in discrete space
            discrete_xvel = self.rng.randint(-max_discrete_xvel, max_discrete_xvel + 1)
            discrete_yvel = self.rng.randint(-max_discrete_yvel, max_discrete_yvel + 1)
            discrete_zvel = self.rng.randint(-max_discrete_zvel, max_discrete_zvel + 1)
            
            # Sample initial rotation velocity in discrete space
            # For rotation, we'll use half the maximum in each direction
            discrete_rotation_vel = self.rng.randint(-max_discrete_angular//2, max_discrete_angular//2 + 1)
            
            # Leave out chunks of latent regions during training
            if self.train and self.exclude_latent_regions:
                # Sample x velocity from restricted domain
                x_region = self.rng.choice([0, 1, 2])  # 0: left high, 1: middle, 2: right high
                if x_region == 0:
                    discrete_xvel = self.rng.randint(-max_discrete_xvel, -max_discrete_xvel//2)
                elif x_region == 1:
                    discrete_xvel = self.rng.randint(-max_discrete_xvel//4, max_discrete_xvel//4 + 1)
                else:  # x_region == 2
                    discrete_xvel = self.rng.randint(max_discrete_xvel//2, max_discrete_xvel + 1)
                
                # Sample y velocity from restricted domain (excluding low velocities)
                y_region = self.rng.choice([0, 1])  # 0: negative high, 1: positive high
                if y_region == 0:
                    discrete_yvel = self.rng.randint(-max_discrete_yvel, -max_discrete_yvel//4)
                else:  # y_region == 1
                    discrete_yvel = self.rng.randint(max_discrete_yvel//4, max_discrete_yvel + 1)
                
                # Sample z velocity from restricted domain (excluding low velocities)
                z_region = self.rng.choice([0, 1])  # 0: negative high, 1: positive high
                if z_region == 0:
                    discrete_zvel = self.rng.randint(-max_discrete_zvel, -max_discrete_zvel//4)
                else:  # z_region == 1
                    discrete_zvel = self.rng.randint(max_discrete_zvel//4, max_discrete_zvel + 1)
            
            # Set initial continuous velocities (for labels)
            xvel_initial = discrete_xvel * xvel_step * max_vel
            yvel_initial = discrete_yvel * yvel_step * max_vel
            zvel_initial = discrete_zvel * zvel_step * max_zvel
            rotation_speed = discrete_rotation_vel * 60.0 / max_discrete_angular
            
            # Set rotation direction label based on rotation velocity
            if discrete_rotation_vel > 0:
                labels['rotation_dir'][i] = 1  # CCW
            elif discrete_rotation_vel < 0:
                labels['rotation_dir'][i] = 2  # CW
            else:
                labels['rotation_dir'][i] = 0  # Constant
            
            # Define base padding in discrete space - minimum safe distance from edges
            base_padding_discrete = max(2, xpos_levels // 20)  # At least 2 or 5% of width
            
            # Sample initial positions in discrete space with appropriate padding
            # We'll calculate the initial padding based on maximum scale to ensure safety
            initial_padding = int(base_padding_discrete * max_scale / min_scale)
            
            # Make sure we don't start too close to the boundary
            discrete_posX = self.rng.randint(initial_padding, xpos_levels - initial_padding)
            discrete_posY = self.rng.randint(initial_padding, ypos_levels - initial_padding)
            discrete_scale = self.rng.randint(0, zpos_levels)
            discrete_rotation = self.rng.randint(0, theta_levels)
            
            # Convert to normalized space
            norm_posX = -1.0 + discrete_posX * xpos_step
            norm_posY = -1.0 + discrete_posY * ypos_step
            norm_scale = discrete_scale * zpos_step
            norm_rotation = discrete_rotation * theta_step
            
            # Convert to pixel space (for physics)
            posX = (norm_posX + 1.0) * (w / 2)
            posY = (norm_posY + 1.0) * (h / 2)
            current_scale = norm_scale * (max_scale - min_scale) + min_scale
            initial_rotation = norm_rotation * 360.0
            
            # Calculate and store speed (magnitude of velocity)
            labels['speed'][i] = np.sqrt(xvel_initial**2 + yvel_initial**2)
            
            # Store angular speed (magnitude)
            labels['angular_speed'][i] = abs(rotation_speed)
            
            # Generate sequence of transformations
            sequence = []
            
            # Track current discrete positions and velocities
            current_discrete_posX = discrete_posX
            current_discrete_posY = discrete_posY
            current_discrete_scale = discrete_scale
            current_discrete_rotation = discrete_rotation
            
            current_discrete_xvel = discrete_xvel
            current_discrete_yvel = discrete_yvel
            current_discrete_zvel = discrete_zvel
            
            # Generate each frame's transformation
            for t in range(self.seq_len):
                # Store the previous position before updating
                previous_discrete_posX = current_discrete_posX
                previous_discrete_posY = current_discrete_posY
                
                if t > 0:
                    # Calculate adaptive padding based on current scale
                    # Higher scale = bigger sprite = need more padding
                    current_norm_scale = current_discrete_scale * zpos_step
                    current_scale_ratio = min_scale + current_norm_scale * (max_scale - min_scale)
                    current_padding = int(base_padding_discrete * current_scale_ratio / min_scale)
                    
                    # First check if the next step will cross boundary
                    next_discrete_posX = current_discrete_posX + current_discrete_xvel
                    next_discrete_posY = current_discrete_posY + current_discrete_yvel
                    
                    # X boundary check - BEFORE updating position
                    boundary_x_violated = False
                    if next_discrete_posX >= xpos_levels - current_padding:
                        # Would hit right wall
                        boundary_x_violated = True
                        current_discrete_xvel = -abs(current_discrete_xvel)
                        labels['xdir'][i, t] = 2  # Now moving left
                    elif next_discrete_posX < current_padding:
                        # Would hit left wall
                        boundary_x_violated = True
                        current_discrete_xvel = abs(current_discrete_xvel)
                        labels['xdir'][i, t] = 1  # Now moving right
                    else:
                        # No collision
                        if current_discrete_xvel > 0:
                            labels['xdir'][i, t] = 1  # Moving right
                        elif current_discrete_xvel < 0:
                            labels['xdir'][i, t] = 2  # Moving left
                        else:
                            labels['xdir'][i, t] = 0  # Not moving in x
                    
                    # Y boundary check - BEFORE updating position
                    boundary_y_violated = False
                    if next_discrete_posY >= ypos_levels - current_padding:
                        # Would hit bottom wall
                        boundary_y_violated = True
                        current_discrete_yvel = -abs(current_discrete_yvel)
                        labels['ydir'][i, t] = 1  # Now moving up
                    elif next_discrete_posY < current_padding:
                        # Would hit top wall
                        boundary_y_violated = True
                        current_discrete_yvel = abs(current_discrete_yvel)
                        labels['ydir'][i, t] = 2  # Now moving down
                    else:
                        # No collision
                        if current_discrete_yvel > 0:
                            labels['ydir'][i, t] = 2  # Moving down
                        elif current_discrete_yvel < 0:
                            labels['ydir'][i, t] = 1  # Moving up
                        else:
                            labels['ydir'][i, t] = 0  # Not moving in y
                    
                    # Now update positions with boundary reflection if needed
                    if boundary_x_violated:
                        # If we would cross boundary, reflect position from boundary
                        if next_discrete_posX >= xpos_levels - current_padding:
                            # Reflect from right boundary
                            current_discrete_posX = 2 * (xpos_levels - current_padding) - next_discrete_posX
                        else:
                            # Reflect from left boundary
                            current_discrete_posX = 2 * current_padding - next_discrete_posX
                    else:
                        # Normal update if no boundary violation
                        current_discrete_posX += current_discrete_xvel
                    
                    if boundary_y_violated:
                        # If we would cross boundary, reflect position from boundary
                        if next_discrete_posY >= ypos_levels - current_padding:
                            # Reflect from bottom boundary
                            current_discrete_posY = 2 * (ypos_levels - current_padding) - next_discrete_posY
                        else:
                            # Reflect from top boundary
                            current_discrete_posY = 2 * current_padding - next_discrete_posY
                    else:
                        # Normal update if no boundary violation
                        current_discrete_posY += current_discrete_yvel
                    
                    # Update scale position and handle scale boundary
                    next_discrete_scale = current_discrete_scale + current_discrete_zvel
                    
                    # Z (scale) boundary checks
                    if next_discrete_scale >= zpos_levels - 1:
                        # Would hit max scale
                        current_discrete_zvel = -abs(current_discrete_zvel)
                        current_discrete_scale = 2 * (zpos_levels - 1) - next_discrete_scale
                        labels['zdir'][i, t] = 2  # Now decreasing
                    elif next_discrete_scale <= 0:
                        # Would hit min scale
                        current_discrete_zvel = abs(current_discrete_zvel)
                        current_discrete_scale = -next_discrete_scale  # Reflect from 0
                        labels['zdir'][i, t] = 1  # Now increasing
                    else:
                        # No collision - normal update
                        current_discrete_scale = next_discrete_scale
                        # Set direction label
                        if current_discrete_zvel > 0:
                            labels['zdir'][i, t] = 1  # Increasing
                        elif current_discrete_zvel < 0:
                            labels['zdir'][i, t] = 2  # Decreasing
                        else:
                            labels['zdir'][i, t] = 0  # Not changing scale
                    
                    # Update rotation
                    current_discrete_rotation = (current_discrete_rotation + discrete_rotation_vel) % theta_levels
                else:
                    # For t=0, set initial direction labels
                    if current_discrete_xvel > 0:
                        labels['xdir'][i, t] = 1  # Moving right
                    elif current_discrete_xvel < 0:
                        labels['xdir'][i, t] = 2  # Moving left
                    else:
                        labels['xdir'][i, t] = 0  # Not moving in x
                        
                    if current_discrete_yvel > 0:
                        labels['ydir'][i, t] = 2  # Moving down
                    elif current_discrete_yvel < 0:
                        labels['ydir'][i, t] = 1  # Moving up
                    else:
                        labels['ydir'][i, t] = 0  # Not moving in y
                        
                    if current_discrete_zvel > 0:
                        labels['zdir'][i, t] = 1  # Increasing
                    elif current_discrete_zvel < 0:
                        labels['zdir'][i, t] = 2  # Decreasing
                    else:
                        labels['zdir'][i, t] = 0  # Not changing scale
                
                # Ensure all discrete values stay within valid bounds
                # This is a safety check to catch any edge cases or bugs
                current_discrete_posX = np.clip(current_discrete_posX, 0, xpos_levels - 1)
                current_discrete_posY = np.clip(current_discrete_posY, 0, ypos_levels - 1)
                current_discrete_scale = np.clip(current_discrete_scale, 0, zpos_levels - 1)
                
                # Convert from discrete to normalized space for labels
                norm_posX = -1.0 + current_discrete_posX * xpos_step
                norm_posY = -1.0 + current_discrete_posY * ypos_step
                norm_scale = current_discrete_scale * zpos_step
                norm_rotation = (current_discrete_rotation * theta_step) % 1.0
                
                # Convert to original units for storing transformation
                current_posX = (norm_posX + 1.0) * (w / 2)
                current_posY = (norm_posY + 1.0) * (h / 2)
                current_scale = norm_scale * (max_scale - min_scale) + min_scale
                current_rotation = norm_rotation * 360.0
                
                # Store current transformation
                transform = {
                    'position': (current_posX, current_posY),
                    'rotation': current_rotation,
                    'scale': current_scale
                }
                sequence.append(transform)
                
                # Store current normalized positions and velocities for labels
                labels['xpos'][i, t] = norm_posX
                labels['ypos'][i, t] = norm_posY
                labels['zpos'][i, t] = norm_scale
                labels['theta'][i, t] = norm_rotation
                
                # FIX: Properly normalize velocities to full [-1, 1] range
                # For discrete velocities, we need to map from the discrete range to the full [-1, 1] range
                # Normalize velocities to [-1, 1] by dividing by the maximum discrete velocity
                norm_xvel = current_discrete_xvel / max_discrete_xvel if max_discrete_xvel > 0 else 0
                norm_yvel = current_discrete_yvel / max_discrete_yvel if max_discrete_yvel > 0 else 0
                norm_zvel = current_discrete_zvel / max_discrete_zvel if max_discrete_zvel > 0 else 0
                
                # Store normalized velocities (now properly in range [-1, 1])
                labels['xvel'][i, t] = norm_xvel
                labels['yvel'][i, t] = norm_yvel
                labels['zvel'][i, t] = norm_zvel
            
            trajectories.append((sprite_idx, sequence))
        
        return trajectories, labels

    def _generate_trajectories(self):
        """
        Generate trajectories for sprites based on discretization setting.
        
        Returns:
            trajectories (list): List of transformation sequences
            labels (dict): Dictionary of trajectory labels
        """
        if self.discretize_latents:
            return self._generate_discrete_trajectories()
        else:
            return self._generate_continuous_trajectories()
    
    def _apply_transform(self, img, transform_params):
        """
        Apply position, rotation, and scale transformations to an image using Kornia.
        This version works with older Kornia versions by manually handling centering.
        
        Args:
            img (torch.Tensor): Input grayscale+alpha tensor [2, H, W]
            transform_params (Dict): Dictionary with transformation parameters
                    
        Returns:
            torch.Tensor: Transformed image
        """
        # Extract transformation parameters
        position = transform_params.get('position', (0, 0))
        rotation = transform_params.get('rotation', 0)
        scale = transform_params.get('scale', 1.0)
        
        # Add batch dimension for Kornia
        img = img.unsqueeze(0)  # [1, 2/4, H, W]
        
        # Get original image dimensions and output dimensions
        _, n_channels, img_h, img_w = img.shape
        out_h, out_w = self.output_size
        
        # Calculate the center of the sprite
        center_x, center_y = img_w / 2, img_h / 2
        
        # Import torch locally to ensure it's available
        import torch
        
        # Convert rotation to radians
        angle_rad = rotation * (np.pi / 180.0)
        cos_a = torch.cos(torch.tensor(angle_rad, device=self.device))
        sin_a = torch.sin(torch.tensor(angle_rad, device=self.device))
        
        # Desired position of the sprite center
        tx, ty = position
        
        # Create the transformation matrix [2, 3]
        transform_matrix = torch.zeros(2, 3, device=self.device)
        
        # Set rotation and scaling components
        transform_matrix[0, 0] = cos_a * scale
        transform_matrix[0, 1] = -sin_a * scale
        transform_matrix[1, 0] = sin_a * scale
        transform_matrix[1, 1] = cos_a * scale
        
        # Calculate translation to place the sprite's center at the desired position
        # This formula compensates for the rotation and scaling around the center
        transform_matrix[0, 2] = tx - (center_x * cos_a * scale - center_y * sin_a * scale)
        transform_matrix[1, 2] = ty - (center_x * sin_a * scale + center_y * cos_a * scale)
        
        # Add batch dimension
        transform_matrix = transform_matrix.unsqueeze(0)  # [1, 2, 3]
        
        # Apply the transformation
        transformed_img = K.geometry.transform.warp_affine(
            img, 
            transform_matrix, 
            dsize=self.output_size,
            align_corners=True
        )
        
        return transformed_img

    def _batch_apply_transform(self, imgs, positions, rotations, scales):
        """
        Apply position, rotation, and scale transformations to a batch of images in parallel.
        This version works with older Kornia versions by manually handling centering.
        
        Args:
            imgs (torch.Tensor): Batch of input grayscale+alpha tensors [B, 2, H, W]
            positions (torch.Tensor): Tensor of (x, y) positions [B, 2]
            rotations (torch.Tensor): Tensor of rotation angles in degrees [B]
            scales (torch.Tensor): Tensor of scale factors [B]
                    
        Returns:
            torch.Tensor: Batch of transformed images [B, 2, H, W]
        """
        batch_size = imgs.shape[0]
        n_channels, img_h, img_w = imgs.shape[1:]
        out_h, out_w = self.output_size
        
        # Import torch and math locally
        import torch
        import math
        
        # Calculate the center of the sprite
        center_x, center_y = img_w / 2, img_h / 2
        
        # Convert rotations to radians
        angles_rad = rotations * (math.pi / 180.0)
        cos_angles = torch.cos(angles_rad)
        sin_angles = torch.sin(angles_rad)
        
        # Create transformation matrices for the entire batch [B, 2, 3]
        transform_matrices = torch.zeros(batch_size, 2, 3, device=self.device)
        
        # Set rotation and scaling components
        transform_matrices[:, 0, 0] = cos_angles * scales
        transform_matrices[:, 0, 1] = -sin_angles * scales
        transform_matrices[:, 1, 0] = sin_angles * scales
        transform_matrices[:, 1, 1] = cos_angles * scales
        
        # Calculate translations to place the sprite centers at the desired positions
        # This formula compensates for the rotation and scaling around the center
        transform_matrices[:, 0, 2] = positions[:, 0] - (center_x * cos_angles * scales - center_y * sin_angles * scales)
        transform_matrices[:, 1, 2] = positions[:, 1] - (center_x * sin_angles * scales + center_y * cos_angles * scales)
        
        # Apply the transformations
        transformed_imgs = K.geometry.transform.warp_affine(
            imgs,
            transform_matrices,
            dsize=self.output_size,
            align_corners=True
        )
        
        return transformed_imgs
            
    def _create_video(self, sprite_idx, transform_sequence):
        """
        Create a video by applying a sequence of transformations to a sprite in parallel
        
        Args:
            sprite_idx (int): Index of sprite to use
            transform_sequence (List[Dict]): Sequence of transformation parameters
            
        Returns:
            torch.Tensor: Video as a tensor with shape [T, 1, H, W]
        """
        # Get the sprite and prepare batch of identical sprites
        img = self.sprites[sprite_idx]  # [2, H, W]
        seq_len = len(transform_sequence)
        
        # Generate all backgrounds for this video at once using fully vectorized approach
        backgrounds = self._create_backgrounds_for_video(seq_len)
        
        # Extract all transformation parameters into separate lists
        positions_x = [params['position'][0] for params in transform_sequence]
        positions_y = [params['position'][1] for params in transform_sequence]
        positions = torch.tensor(list(zip(positions_x, positions_y)), dtype=torch.float32, device=self.device)
        rotations = torch.tensor([params['rotation'] for params in transform_sequence], dtype=torch.float32, device=self.device)
        scales = torch.tensor([params['scale'] for params in transform_sequence], dtype=torch.float32, device=self.device)
        
        # Prepare batch of identical sprites
        batch_img = img.unsqueeze(0).repeat(seq_len, 1, 1, 1)  # [seq_len, 2, H, W]
        
        # Apply transformations in parallel for each sequence step
        transformed_imgs = self._batch_apply_transform(batch_img, positions, rotations, scales)
        
        if self.grayscale:
            # Extract grayscale and alpha channels
            gray = transformed_imgs[:, 0:1]  # [seq_len, 1, H, W]
            alpha = transformed_imgs[:, 1:2]  # [seq_len, 1, H, W]
        else:
            gray = transformed_imgs[:, 0:3]  # [seq_len, 3, H, W]
            alpha = transformed_imgs[:, 3:4]  # [seq_len, 1, H, W]
        
        # Composite using alpha blending for all frames at once
        # result = alpha * foreground + (1 - alpha) * background
        result = alpha * gray + (1 - alpha) * backgrounds

        # Randomly occlude some consecutive frames
        # Ensure we have enough frames to occlude
        if seq_len <= self.occlude_n_frames + 4 and self.occlude_n_frames > 0:
            raise ValueError(f"Sequence too short (seq_len={seq_len}) for occlusion of {self.occlude_n_frames} frames; need > occlude_n_frames + 4")
        occlusion_start = self.rng.randint(2, seq_len - self.occlude_n_frames - 2)
        occlusion_end = occlusion_start + self.occlude_n_frames
        # Replace occluded frames with backgrounds
        result[occlusion_start:occlusion_end] = backgrounds[occlusion_start:occlusion_end]

        # add noise on top of the video
        if self.noise_on_top:
            noise = self._generate_noise_for_video(seq_len)
            # add noise to the video and ensure it doesn't exceed [0, 1] range
            result = torch.clamp(result + noise, 0.0, 1.0)
        
        return result
        
    def __len__(self):
        """Return the number of sequences in the split"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a synthetic video sample with associated labels
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - video (torch.Tensor): Video tensor with shape [T, C, H, W] where C=1 for grayscale
                - labels (tuple): Tuple containing various label tensors
        """
        # Map idx to the actual index in trajectories
        traj_idx = self.indices[idx]
        
        # Get the sprite index and transformation sequence
        sprite_idx, transform_sequence = self.trajectories[traj_idx]
        
        # Create the video using the optimized method
        video = self._create_video(sprite_idx, transform_sequence)

        # Apply normalization using the computed or provided values
        if self.grayscale:
            video = (video - self.mean) / self.std
        else:
            video = (video - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        
        # Collect labels for this trajectory
        traj_labels = {}
        for key, value in self.labels.items():
            if value.ndim == 1:
                traj_labels[key] = torch.tensor(value[traj_idx])
            else:
                traj_labels[key] = torch.tensor(value[traj_idx, :])
        
        # Get discretization levels from the configuration if available
        xpos_levels = self.discretization_config['xpos']['levels'] if 'xpos' in self.discretization_config else 33
        ypos_levels = self.discretization_config['ypos']['levels'] if 'ypos' in self.discretization_config else 33
        zpos_levels = self.discretization_config['zpos']['levels'] if 'zpos' in self.discretization_config else 17
        theta_levels = self.discretization_config['theta']['levels'] if 'theta' in self.discretization_config else 36
        
        # Convert rotation angle to a discrete label and sin/cos pair
        theta = torch.tensor(self.labels['theta'][traj_idx], dtype=torch.float32)
        sin_theta = torch.sin(theta * 2 * np.pi)
        cos_theta = torch.cos(theta * 2 * np.pi)
        
        # Discretization must match exactly how continuous values are converted in trajectory generation
        # From discrete trajectory generation:
        # norm_posX = -1.0 + discrete_posX * xpos_step
        # Where xpos_step = 2.0 / (xpos_levels - 1)
        
        # To invert this for positions:
        # discrete_posX = (norm_posX + 1.0) / xpos_step = (norm_posX + 1.0) * (xpos_levels - 1) / 2.0
        xpos_discrete = ((traj_labels['xpos'] + 1.0) * (xpos_levels - 1) / 2.0).round().to(torch.long)
        ypos_discrete = ((traj_labels['ypos'] + 1.0) * (ypos_levels - 1) / 2.0).round().to(torch.long)
        
        # For scale and rotation:
        # norm_scale = discrete_scale * zpos_step
        # Where zpos_step = 1.0 / (zpos_levels - 1)
        
        # To invert:
        # discrete_scale = norm_scale / zpos_step = norm_scale * (zpos_levels - 1)
        zpos_discrete = (traj_labels['zpos'] * (zpos_levels - 1)).round().to(torch.long)
        theta_discrete = (theta * (theta_levels - 1)).round().to(torch.long) % theta_levels
        
        # Apply clipping to ensure values stay within valid range
        xpos_discrete = torch.clamp(xpos_discrete, 0, xpos_levels - 1)
        ypos_discrete = torch.clamp(ypos_discrete, 0, ypos_levels - 1)
        zpos_discrete = torch.clamp(zpos_discrete, 0, zpos_levels - 1)
        
        # Organize labels in the format expected by the training code
        # Converting to the same format as dSprites dataset
        seq_labels = torch.tensor([
            self.labels['sprite_idx'][traj_idx],
            self.labels['rotation_dir'][traj_idx]
        ], dtype=torch.long)
        
        dense_labels = torch.stack([
            torch.tensor(self.labels['xdir'][traj_idx], dtype=torch.long),
            torch.tensor(self.labels['ydir'][traj_idx], dtype=torch.long),
            torch.tensor(self.labels['zdir'][traj_idx], dtype=torch.long),
            xpos_discrete,
            ypos_discrete,
            zpos_discrete,
            theta_discrete,
        ], dim=1)
        
        cts_labels = torch.tensor([
            self.labels['speed'][traj_idx],
            self.labels['angular_speed'][traj_idx]
        ], dtype=torch.float32)
        
        cts_dense_labels = torch.stack([
            torch.tensor(self.labels['xpos'][traj_idx], dtype=torch.float32),
            torch.tensor(self.labels['ypos'][traj_idx], dtype=torch.float32),
            torch.tensor(self.labels['zpos'][traj_idx], dtype=torch.float32),
            torch.tensor(self.labels['xvel'][traj_idx], dtype=torch.float32),
            torch.tensor(self.labels['yvel'][traj_idx], dtype=torch.float32),
            torch.tensor(self.labels['zvel'][traj_idx], dtype=torch.float32),
            sin_theta,
            cos_theta,
        ], dim=1)
        
        aux_labels = torch.stack([
            theta,
        ], dim=1)

        return video, (seq_labels, dense_labels, cts_labels, cts_dense_labels, aux_labels)

def save_videos_as_gifs_and_pdfs(videos, labels, output_dir="video_gifs", fps=10, max_videos=8, colormap=None, pointer=True, mean=None, std=None):
    """
    Save a batch of videos as both GIF animations and PDF files.
    Images are denormalized, inverted, and a green pointer and path trace are added based on object position from labels.
    The PDF saves only the first 8 frames horizontally with black borders around each frame, 
    and velocity plots are saved as separate files.
    
    Args:
        videos (torch.Tensor): Batch of videos with shape [B, T, C, H, W] 
                              where B=batch, T=time, C=channels, H=height, W=width
        labels (tuple): Tuple of label tensors from the dataset
                       (seq_labels, dense_labels, cts_labels, cts_dense_labels)
        output_dir (str): Directory to save the GIFs/PDFs (will be created if it doesn't exist)
        fps (int): Frames per second for the GIF animation
        max_videos (int): Maximum number of videos to save
        colormap (str, optional): Matplotlib colormap name to apply
        pointer (bool): Whether to add position pointer and path trace
        mean (float, optional): Mean value used for normalization, needed for denormalization
        std (float, optional): Standard deviation used for normalization, needed for denormalization
    
    Returns:
        list: Paths to the saved GIF and PDF files
    """
    import imageio
    import os
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import matplotlib.cm as cm
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches
    import seaborn as sns
    import matplotlib.path as mpath
    import matplotlib.lines as mlines

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure videos is on CPU and convert to numpy
    if isinstance(videos, torch.Tensor):
        videos = videos.detach().cpu().numpy()
        mean = mean.detach().cpu().numpy() if mean is not None else None
        std = std.detach().cpu().numpy() if std is not None else None
    
    # Limit the number of videos to save
    num_videos = min(videos.shape[0], max_videos)
    saved_paths = []
    
    # Extract label information
    label_info = []
    seq_labels, dense_labels, cts_labels, cts_dense_labels, aux_labels = labels
    
    # Convert to CPU if they're torch tensors
    if isinstance(seq_labels, torch.Tensor):
        seq_labels = seq_labels.detach().cpu().numpy()
    if isinstance(dense_labels, torch.Tensor):
        dense_labels = dense_labels.detach().cpu().numpy()
    if isinstance(cts_labels, torch.Tensor):
        cts_labels = cts_labels.detach().cpu().numpy()
    if isinstance(cts_dense_labels, torch.Tensor):
        cts_dense_labels = cts_dense_labels.detach().cpu().numpy()
    
    # Extract useful label info for filenames
    for i in range(num_videos):
        # Get sprite index from seq_labels[i, 0]
        sprite_idx = int(seq_labels[i, 0])
        
        # Get rotation direction from seq_labels[i, 1]
        # 0=constant, 1=ccw, 2=cw (as defined in _generate_trajectories)
        rotation_dir = ["constant", "ccw", "cw"][int(seq_labels[i, 1])]
        
        # Speed and angular speed from cts_labels
        speed = float(cts_labels[i, 0])
        angular_speed = float(cts_labels[i, 1])
        
        # Create label info string
        info = f"sprite{sprite_idx}_rot-{rotation_dir}_speed-{speed:.1f}_ang-{angular_speed:.1f}"
        label_info.append(info)
    
    # Process each video
    for i in range(num_videos):
        # Get the video frames
        video = videos[i]  # [T, C, H, W]
        
        # Get the number of frames, channels, height, width
        num_frames, num_channels, height, width = video.shape

        # Denormalize the video if mean and std are provided
        if mean is not None and std is not None:
            if num_channels == 1:
                video = video * std + mean
            else:
                video = video * std[:, None, None] + mean[:, None, None]
        
        # Create lists to store the processed frames
        frames = []
        pil_frames = []  # For PDF
        
        # Calculate object positions for this video
        positions = []
        velocities_x = []
        velocities_y = []
        
        # Get width and height for denormalization
        h, w = height, width
        
        # Extract positions from cts_dense_labels tensor
        # We have per-frame position data [B, T, 6]
        prev_xpos = None
        prev_ypos = None
        
        for t in range(num_frames):
            if t < cts_dense_labels.shape[1]:
                # Get normalized positions for this frame
                norm_xpos = cts_dense_labels[i, t, 0].item()  # Convert tensor to scalar
                norm_ypos = cts_dense_labels[i, t, 1].item()  # Convert tensor to scalar
                
                # Extract velocities
                norm_xvel = cts_dense_labels[i, t, 3].item()  # x velocity
                norm_yvel = cts_dense_labels[i, t, 4].item()  # y velocity
                velocities_x.append(norm_xvel)
                velocities_y.append(norm_yvel)
            else:
                # If we don't have data for all frames, use the last available
                norm_xpos = cts_dense_labels[i, -1, 0].item()
                norm_ypos = cts_dense_labels[i, -1, 1].item()
                
                # For velocities too
                velocities_x.append(velocities_x[-1])
                velocities_y.append(velocities_y[-1])
            
            # Denormalize to pixel coordinates
            xpos = (norm_xpos * (w/2)) + (w/2)
            ypos = (norm_ypos * (h/2)) + (h/2)
            positions.append((xpos, ypos))
        
        # Process each frame
        for t in range(num_frames):
            # Get the frame
            frame = video[t]  # [C, H, W]
            
            # Normalize to [0, 1] range if needed for display
            if frame.min() < 0 or frame.max() > 1:
                frame = np.clip(frame, 0, 1)
            
            # Convert to 8-bit for display
            frame = (frame * 255).astype(np.uint8)
            
            # Reshape grayscale frame if needed
            if num_channels == 1:
                frame = frame.reshape(height, width)  # [H, W]
                
                # Apply colormap if specified
                if colormap is not None:
                    # Normalize to 0-1 for colormap
                    frame_norm = frame.astype(float) / 255.0
                    # Apply colormap (using updated API)
                    colored = plt.colormaps.get_cmap(colormap)(frame_norm)
                    # Convert to 0-255 uint8
                    frame = (colored[:, :, :3] * 255).astype(np.uint8)
                    # Invert the colored image
                    frame = 255 - frame
                else:
                    # Invert the grayscale image
                    frame = 255 - frame
            else:
                # For RGB or other multi-channel formats, transpose to [H, W, C]
                frame = np.transpose(frame, (1, 2, 0))
                # # Invert the color image
                # frame = 255 - frame
            
            # Create a figure and axis for plotting the trajectory with higher resolution
            # Increase resolution by using a higher DPI
            upscale_factor = 4  # Upscale by 4x for smoother curves
            fig = plt.figure(figsize=(width/100, height/100), dpi=100 * upscale_factor)
            ax = fig.add_axes([0, 0, 1, 1])
            
            # Show the frame without interpolation, but allow it to be upscaled
            ax.imshow(frame, interpolation='none')
            
            # Add green pointer and path if pointer is enabled
            if pointer:
                # Draw path up to current frame with the specified color
                if t > 0:
                    x_positions = [pos[0] for pos in positions[:t+1]]
                    y_positions = [pos[1] for pos in positions[:t+1]]
                    # Use a smoother path representation with antialiasing and higher resolution
                    ax.plot(x_positions, y_positions, color='gray', linewidth=1.0, 
                            solid_capstyle='round', solid_joinstyle='round', antialiased=True)
                
                # # Draw current position pointer
                # xpos, ypos = positions[t]
                # pointer_radius = 1
                
                # # Draw white outline - scaled for higher resolution
                # circle_outline = plt.Circle((xpos, ypos), (pointer_radius + 1) /2, 
                #                             color='white', fill=False, linewidth=0.5/2)
                # ax.add_patch(circle_outline)
                
                # # Draw colored circle - scaled for higher resolution
                # circle = plt.Circle((xpos, ypos), pointer_radius/2, color='white')
                # ax.add_patch(circle)
            
            # Remove axis and set limits
            ax.axis('off')
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)  # Flipped to match image coordinates
            
            # Convert the figure to an image
            fig.canvas.draw()
            frame_with_plot = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            
            # Convert to PIL image
            pil_img = Image.fromarray(frame_with_plot)
            
            # Append to frames list for GIF
            frames.append(pil_img)
            # Save a copy for PDF
            pil_frames.append(pil_img.copy())
        
        # Create base filename
        base_filename = f"video_{i}_{label_info[i]}"
        
        # Save as GIF
        gif_path = os.path.join(output_dir, f"{base_filename}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        saved_paths.append(gif_path)
        
        # Save as PDF with only the first 8 frames horizontally with black borders
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        with PdfPages(pdf_path) as pdf:
            # Determine how many frames to include (minimum of 8 or actual frame count)
            frames_to_include = min(8, len(pil_frames))
            
            # Create a figure that will contain all frames in a row
            # Set exact width to 3 inches as requested
            total_width = 3  # inches
            
            # Width of each frame (accounting for spacing)
            frame_width = total_width / frames_to_include
            frame_height = frame_width * (height / width)  # maintain aspect ratio
            
            # Create figure for the frames with fixed width
            fig = plt.figure(figsize=(total_width, frame_height))
            
            # Create a grid layout with just one row and multiple columns
            # Remove spacing between frames
            gs = GridSpec(1, frames_to_include, figure=fig, wspace=0)
            
            # Add each frame to the grid with black borders
            for idx in range(frames_to_include):
                ax = fig.add_subplot(gs[0, idx])
                ax.imshow(np.array(pil_frames[idx]))
                
                # Add thinner black border around each frame
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.1)  # Reduced from 1 to 0.1
                    spine.set_visible(True)
                
                # Remove ticks and labels
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Adjust layout and save
            pdf.savefig(fig, bbox_inches='tight', dpi=300, transparent=True)
            plt.close(fig)
        
        saved_paths.append(pdf_path)
        
        # Save velocity plot as a separate file
        vel_path = os.path.join(output_dir, f"{base_filename}_velocity.pdf")
        
        # Create the figure for velocities with exact dimensions as requested
        plt.figure(figsize=(1.5, 1.0))
        
        # Create time axis
        time_points = list(range(num_frames))
        
        # Plot velocities
        plt.plot(time_points, velocities_x, color='#BB5566', label=r'$\dot{x}$')
        plt.plot(time_points, velocities_y, color='#004488', label=r'$\dot{y}$')
        
        # Set custom y-ticks at -1 and 1 only
        plt.yticks([-1, 1], ['-1', '1'])
        
        # Set custom x-ticks at only 0 and 30
        plt.xticks([0, 30])
        
        # Set labels with fontsize 8
        plt.ylabel('Vel.', fontsize=8)
        plt.xlabel('Time', fontsize=8)
        
        # Set legend with fontsize 6 and place it outside to the right
        plt.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
        
        # Set tick label font size
        plt.tick_params(axis='both', which='major', labelsize=8)
        sns.despine()
        
        # Adjust layout to accommodate the legend
        plt.tight_layout()
        
        # Save the velocity plot
        plt.savefig(vel_path, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(vel_path)
        
        print(f"Saved video {i} as GIF: {gif_path}")
        print(f"Saved video {i} as PDF: {pdf_path} (first {frames_to_include} frames)")
        print(f"Saved velocity plot: {vel_path}")
    
    return saved_paths


# Example usage
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    import time
    
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    # Create the dataset with fixed noise and grid patterns
    dataset = SpriteVideoDataset(
        data_dir=data_dir,
        sprite_img_dir="animals",
        split="train",
        output_size=(64, 64),
        seq_len=32,
        num_sequences=1000,
        seed=42,
        device=device,
        max_sprites=8,
        noise_type=None, # Options: 'salt_pepper', 'gaussian'
        noise_intensity=0.1, # Intensity of noise (0.0 to 1.0)
        discretize_latents=True, # Whether to discretize latent variables
        freeze_noise=False, # Whether to keep noise pattern constant across frames
        grid_enabled=False, # Whether to add grid pattern
        freeze_grid=True,   # Use different grid patterns for each frame
        background=1.0, # Background color (1.0 for white, 0.0 for black)
        occlude_n_frames=4, # Number of frames to occlude in each video
    )
    
    num_workers = 0
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None
    )
    
    mean, std = dataset.mean, dataset.std
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="animal_videos",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
        mean=mean,
        std=std
    )
    # Iterate through the dataloader with proper exception handling
    # try:
    #     for batch_idx, (batch_videos, batch_labels) in enumerate(dataloader):
    #         # # Move batch to GPU if needed for downstream processing
    #         # if device == 'cuda':
    #         #     batch_videos = batch_videos.cuda(non_blocking=True)
    #         #     batch_labels = tuple(label.cuda(non_blocking=True) for label in batch_labels)
            
            # # Unpack labels
            # seq_labels, dense_labels, cts_labels, cts_dense_labels = batch_labels
            
            # print(f"Batch {batch_idx}, videos shape: {batch_videos.shape}")
            # print(f"Sequence labels shape: {seq_labels.shape}")
            
            # # Check the ranges of every label
            # for i in range(seq_labels.shape[-1]):
            #     print(f"Label {i}: min={seq_labels[:, i].min()}, max={seq_labels[:, i].max()}")
            # for i in range(dense_labels.shape[-1]):
            #     print(f"Dense label {i}: min={dense_labels[:, :, i].min()}, max={dense_labels[:, :, i].max()}")
            # for i in range(cts_labels.shape[-1]):
            #     print(f"Continuous label {i}: min={cts_labels[:, i].min()}, max={cts_labels[:, i].max()}")
            # for i in range(cts_dense_labels.shape[-1]):
            #     print(f"Continuous dense label {i}: min={cts_dense_labels[:, :, i].min()}, max={cts_dense_labels[:, :, i].max()}")

    #         # # Count sprite usage
    #         # for sprite_idx in seq_labels[:, 2]:  # Sprite index is in the 3rd position
    #         #     sprite_counts[sprite_idx.item()] += 1
            
    #         # # Print performance metrics every 10 batches
    #         # if batch_idx > 0 and batch_idx % 10 == 0:
    #         #     elapsed = time.time() - start_time
    #         #     avg_time_per_batch = elapsed / (batch_idx + 1)
    #         #     fps = batch_videos.size(0) * batch_videos.size(1) / avg_time_per_batch
    #         #     print(f"Average time per batch: {avg_time_per_batch:.4f}s, Total FPS: {fps:.2f}")

    #         if batch_idx >= 100:  # Just process a few batches for the example
    #             break

    # except Exception as e:
    #     print(f"Error during dataloader iteration: {e}")
    # finally:
    #     print(f"Final mean: {mean:.4f}, Final std: {std:.4f}")
    #     # Clean up resources 
    #     print("\nDone")
    #     # print("\nAverage sprite usage statistics:")
    #     # for sprite_idx, count in sprite_counts.items():
    #     #     total_count = sum(sprite_counts.values())
    #     #     if total_count > 0:  # Avoid division by zero
    #     #         print(f"Sprite {sprite_idx}: used {count} times ({count/total_count*100:.1f}%)")