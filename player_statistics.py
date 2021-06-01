

class PlayerStatistics:
    def __init__(self):
        self.episode_lengths = []
        self.episode_rewards = []
        self.video = None

    def log_finished_game(self, length, rewards):
        self.episode_lengths.append(length)
        self.episode_rewards.append(rewards)

    def log_video_frames(self, frames):
        if len(frames) > 10:
            self.video = frames

    def get_statistics(self):
        stats = (self.episode_lengths, self.episode_rewards)
        self.episode_lengths = []
        self.episode_rewards = []
        return stats

    def get_video(self):
        video = self.video
        self.video = None
        return video
