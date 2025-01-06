import clip
from liv import load_liv
from flax.training import checkpoints
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
import transformers
from pathlib import Path

from diffusers import UniPCMultistepScheduler
from .models.pipeline_controlnet import StableDiffusionControlNetPipelineTakSIE
from .models.controlnet import ControlNetModelTakSIE

import torchvision.transforms as T
import torch
import torch.distributions as D
import torch.nn as nn
import jax
import numpy as np
from PIL import Image
import cv2
import copy
import os
import yaml
transform_for_liv = T.Compose([T.ToTensor()])


def add_img_thumbnail_to_imgs(imgs: np.ndarray, img_thumbnails: np.ndarray):
    size = imgs.shape[-2:]
    i_h = int(size[0] / 3)
    i_w = int(size[1] / 3)
    for i, img_thumbnail in enumerate(img_thumbnails):
        resize_goal_img = cv2.resize(
            img_thumbnail, dsize=(i_h, i_w), interpolation=cv2.INTER_CUBIC
        )
        mod_goal_img = np.expand_dims(
            resize_goal_img.transpose(2, 0, 1), axis=0)
        imgs[i:i+1, :, -i_h:, :i_w] = mod_goal_img
    return imgs


class taksie_wrapper():
    def __init__(self, running_config=None, device="cuda:0"):
        self.device = device
        self.running_config = running_config
        if (self.running_config != None):
            # hyper_parameter_tuning
            self.max_per_frame = self.running_config["max_per_frame"]
            self.if_regenerate = self.running_config["if_regenerate"]
            self.current_keyframes = self.running_config["current_keyframes"]
            self.target_keyframes = self.running_config["target_keyframes"]
            self.controlnet_path = self.running_config["controlnet_path"]
            self.unet_path = self.running_config["unet_path"]
            self.random_seeds_for_dgcbc = self.running_config["random_seeds_for_dgcbc"]
            np.random.seed(self.random_seeds_for_dgcbc)
            self.subgoal_distance_threshhold = self.running_config["subgoal_distance_threshhold"]
            self.pix2pix_controlnet = self.running_config["pix2pix_controlnet"]
            self.negative_prompt_yaml_path = self.running_config["negative_prompt_yaml_path"]
            self.guidance_scale = self.running_config["guidance_scale"]
            self.image_guidance_scale = self.running_config["image_guidance_scale"]
            self.num_inference_steps = self.running_config["num_inference_steps"]
            self.resolution = self.running_config["resolution"]
            self.min_per_frame = self.running_config["min_per_frame"]
            self.if_normlization_action = self.running_config["if_normlization_action"]

            self.validation_prompt_path = os.path.join(
                Path(__file__).absolute().parents[0], "new_playtable_validation.yaml")

            if (self.running_config["feature"] == "clip"):
                self.get_img_feature = self.get_clip_feature
            elif (self.running_config["feature"] == "liv"):
                self.get_img_feature = self.get_liv_feature
            else:
                print("ERROR:1")
                exit(0)
        else:
            print("ERROR:2")
            exit(0)

        with open(self.negative_prompt_yaml_path, 'r') as file:
            self.negative_prompt = yaml.safe_load(file)

        with open(self.validation_prompt_path, 'r') as file:
            self.validation_prompt = yaml.safe_load(file)

        self.init_controlnet_model(self.controlnet_path, self.unet_path)
        self.distance_subgoal = 100
        self.imageProcessor = transformers.AutoImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", use_auth_token=False)
        self.visionModel = transformers.CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14").requires_grad_(False).eval().to(self.device)

        self.random_seeds = 0
        self.image_goal = ''
        self.goal_this_task = ""
        self.action_list = []
        self.get_jax_action = self.init_jax_model()
        self.action_counter = 1000
        self.current_goal_pil_img = ""
        self.need_generate_next_keyframe = True
        self.img_list = []
        self.steps_lists = []
        self.keyfreams_lists = []
        self.steps_for_this_eps = -1

        self.global_rollout_id = -1
        self.liv = load_liv(modelid="resnet50")
        self.liv.eval()
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=bool)

    def get_negative_prompt(self, task):
        negative_prompt = self.negative_prompt[task]
        # print(f"{task} : {negative_prompt}")
        return negative_prompt[0]

    def get_task_name(self, prompt):
        for keys in self.validation_prompt.keys():
            # print(self.validation_prompt[keys][0])
            # print(prompt)
            if self.validation_prompt[keys][0] == prompt:
                return keys
        print(f"ERROR : {prompt}")
        exit(0)

    def get_liv_feature(self, rgb_static):
        with torch.no_grad():
            image = transform_for_liv(rgb_static).unsqueeze(0).to(self.device)
            img_embedding = self.liv(input=image, modality="vision")
            img_embedding = img_embedding.unsqueeze(0)
            return img_embedding

    def init_controlnet_model(self, controlnet_path, unet_path):
        controlnet_path = controlnet_path
        base_model_path = unet_path
        controlnet = ControlNetModelTakSIE.from_pretrained(
            controlnet_path, torch_dtype=torch.float32).eval()
        self.pipe = StableDiffusionControlNetPipelineTakSIE.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float32
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config)

        self.saving_pil_frames_list = []
        self.rollout_number = 0
        return True

    def reset_videos(self):
        self.steps_for_this_eps = 0
        self.goal_img_list = []
        self.img_list = []

    def save_video_to_file(self, video_filename: str, images: np.ndarray, fps: int = 15):
        """
        Saves rollout video
        images: np.array, images used to create the video
                shape - seq, channels, height, width
        video_filename: str, path used to saved the video file
        """
        output_video = cv2.VideoWriter(
            video_filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            [images.shape[-2:][1], images.shape[-2:][0]],
        )

        images = np.moveaxis(images, 1, -1)[..., ::-1]
        for img in images:
            output_video.write(img)

        output_video.release()

    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_text_feature(self, text):
        text = clip.tokenize(text)
        text_embedding = self.liv(input=text, modality="text")
        return text_embedding

    def save_videos(self, video_file_name):
        # img_array = np.moveaxis(np.stack(self.img_list), -1, 1)
        img_array = np.stack(self.img_list)
        image_list = []
        img_array = img_array.astype(np.uint8)

        self.save_video_to_file(video_file_name, img_array)

    def generation_evaluator(self, subgoal_list, prompt):
        text_liv_feature = self.get_text_feature(prompt)
        min_id = -1
        min_dis = -1000
        for subgoal_id, image in enumerate(subgoal_list):
            generated_img_feature = self.get_liv_feature(image)
            img_text_value_1 = self.liv.module.sim(
                generated_img_feature, text_liv_feature)
            if (img_text_value_1 < min_dis):
                min_id = subgoal_id
        return subgoal_list[min_id]

    def init_jax_model(self):
        from .gcbc_train_config import get_config
        from .gcbc_data_config import get_config as get_data_config
        config = get_config("gc_ddpm_bc")
        config.encoder = self.running_config["encoder"]
        data_config = get_data_config("all")
        encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
        # act_pred_horizon = config["dataset_kwargs"].get("act_pred_horizon")
        act_pred_horizon = self.running_config["act_pred_horizon"]
        # encoder =
        obs_horizon = config["dataset_kwargs"].get("obs_horizon")
        im_size = 200
        checkpoint_weights_path = self.running_config["dgcbc_path"]
        rng = jax.random.PRNGKey(self.random_seeds_for_dgcbc)
        rng, construct_rng = jax.random.split(rng)

        if act_pred_horizon is not None:
            example_actions = np.zeros(
                (1, act_pred_horizon, 7), dtype=np.float32)
        else:
            example_actions = np.zeros((1, 7), dtype=np.float32)
        if obs_horizon is not None:
            example_obs = {
                "image": np.zeros(
                    (1, obs_horizon, im_size, im_size, 3), dtype=np.uint8
                )
            }
        else:
            example_obs = {
                "image": np.zeros((1, im_size, im_size, 3), dtype=np.uint8)
            }
        example_batch = {
            "observations": example_obs,
            "goals": {
                "image": np.zeros((1, im_size, im_size, 3), dtype=np.uint8)
            },
            "actions": example_actions,
        }
        self.agent = agents[config["agent"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **config["agent_kwargs"],
        )
        action_proprio_metadata = data_config["action_proprio_metadata"]
        self.action_mean = np.array(action_proprio_metadata["action"]["mean"])
        self.action_std = np.array(action_proprio_metadata["action"]["std"])
        self.agent = checkpoints.restore_checkpoint(
            checkpoint_weights_path, self.agent)
        # self.agent = orbax.checkpoint.PyTreeCheckpointer().restore("/scratch/qhv6ku/from_cs_server/git_instructed_pix_pix/susie-calvin-checkpoints/gc_policy", item=self.agent)

        def get_jax_action(self, obs, goal_obs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            action = jax.device_get(
                self.agent.sample_actions(obs, goal_obs, seed=key, argmax=True)
            )
            action = action * self.action_std + self.action_mean
            return action
        return get_jax_action

    def predict_action(self, image_obs: np.ndarray, goal_image: np.ndarray):
        action = self.agent.sample_actions(
            image_obs,
            goal_image,
            seed=jax.random.PRNGKey(42),
            temperature=0.0,
        )
        action = np.array(action.tolist())
        action = action[0:4]
        # Scale action
        if (self.if_normlization_action):
            # action = np.array(self.action_statistics["std"]) * action + np.array(self.action_statistics["mean"])
            action = action * self.action_std + self.action_mean

        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * np.array([[True, True, True, True],
                                                                      [True, True,
                                                                          True, False],
                                                                      [True, True,
                                                                          False, False],
                                                                      [True, False, False, False]], dtype=bool)

        # Add to action buffer
        self.action_buffer[0] = action
        self.action_buffer_mask[0] = np.array(
            [True, True, True, True], dtype=bool)

        # Ensemble temporally to predict action
        action_prediction = np.sum(
            self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)

        # Make gripper action either -1 or 1
        if action_prediction[-1] < 0:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        return action_prediction

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0
        self.action_list = []
        self.current_goal_pil_img = ""
        self.goal_this_task = ""
        self.steps_for_this_eps = 0
        self.global_rollout_id += 1
        self.action_counter = 1000
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=bool)
        self.steps_for_current_keyframe = 0
    def relabel_action(self, action_list):
        if (action_list[-1] < 0):
            return -1
        if (action_list[-1] >= 0):
            return 1

    def get_r3m_feature(self, rgb_static_PIL):
        subgoal_input_r3m = self.r3m_transforms(
            rgb_static_PIL).to(self.r3m.device_ids[0])
        subgoal_input_r3m = subgoal_input_r3m.unsqueeze(0)
        subgoal_output_r3m = self.r3m(subgoal_input_r3m * 255.0)
        return subgoal_output_r3m

    def get_clip_feature(self, pil_image):
        image_transforms = T.Compose(
            [
                T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(512)
            ]
        )
        pil_image = image_transforms(pil_image)

        input = self.imageProcessor(images=pil_image, return_tensors="pt")
        input_features = self.visionModel(input['pixel_values'].to(self.device))[
            0].unsqueeze(0).to(self.device)
        return input_features

    def step(self, obs, goal, step):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (str or dict): The goal as a natural language instruction or dictionary with goal images.

        Returns:
            Predicted action.
        """
        if (self.steps_for_this_eps != 0):
            self.goal_img_list.append(np.asarray(
                self.current_goal_pil_img.resize((200, 200))))
            numpy_obs = np.asarray(T.functional.to_pil_image(
                (obs["rgb_obs"]["rgb_static"].squeeze())).resize((200, 200)))
            numpy_goal = np.asarray(
                self.current_goal_pil_img.resize((200, 200)))
            image_concat = np.vstack((numpy_obs, numpy_goal))
            image_concat = cv2.cvtColor(image_concat, cv2.COLOR_RGB2BGR)
            image_concat = image_concat[:, :, ::-1].transpose((2, 0, 1))
            self.img_list.append(image_concat)

        self.need_generate_next_keyframe = False
        prompt = goal
        task_name = self.get_task_name(prompt)
        negative_prompt = self.get_negative_prompt(task_name)
        # print(f"{step} : {goal}")
        if (step == 0):
            self.rollout_number += 1
            self.feature_list = []
            self.steps_for_current_keyframe = 0
            self.distance_subgoal = 0
            self.current_keyframes = 0

        self.control_image = T.functional.to_pil_image(
            (obs["rgb_obs"]["rgb_static"].squeeze()))
        self.control_image = self.control_image.resize((self.resolution,self.resolution))
        # r3m_feature = self.get_r3m_feature(T.functional.to_pil_image(obs["rgb_obs"]["rgb_static"].squeeze() / 2 + 0.5))
        self.liv_feature_of_current = self.get_liv_feature(
            self.control_image)
        # print(step)
        if (step != 0):
            self.distance_subgoal = 1 - \
                self.liv.module.sim(
                    self.liv_feature_of_current, self.liv_feature_of_goal)
        self.steps_for_current_keyframe += 1
        if ((self.steps_for_current_keyframe > self.max_per_frame and self.steps_for_current_keyframe > self.min_per_frame) or self.distance_subgoal < self.subgoal_distance_threshhold):
            self.steps_for_current_keyframe = 0
            self.need_generate_next_keyframe = True
        self.steps_for_this_eps += 1
        if (step == 0 or self.need_generate_next_keyframe):
            # print("HAHA")
            # print(self.current_keyframes)
            self.need_generate_next_keyframe = False
            self.current_keyframes += 1
            if (self.current_keyframes != 1):
                save_path_for_progress_evaluator = "/scratch/qhv6ku/from_cs_server/generated_result_image_for_paper/for_evaluation"
                current_img_file_name = f"{self.global_rollout_id}_{self.current_keyframes}_0.jpg"
                goal_img_file_name = f"{self.global_rollout_id}_{self.current_keyframes}_1.jpg"
                current_img_file_path = os.path.join(
                    save_path_for_progress_evaluator, current_img_file_name)
                goal_img_file_path = os.path.join(
                    save_path_for_progress_evaluator, goal_img_file_name)
                self.control_image.save(current_img_file_path)
                self.current_goal_pil_img.save(goal_img_file_path)
            # print("IT IS RIGHT")
            self.feature_list.append(
                self.get_img_feature(self.control_image))
            # print(self.feature_list)
            feature = torch.cat(self.feature_list, dim=0)
            # feature = torch.cat([feature, feature], dim = 1)
            if (self.pix2pix_controlnet == "controlnet"):
                feature = torch.cat(
                    [feature, feature, feature], dim=1)
                keyframe_image = self.pipe(
                    prompt=prompt, num_inference_steps=self.num_inference_steps, negative_prompt=negative_prompt, image=self.control_image.resize((self.resolution, self.resolution)), class_label=feature, guidance_scale=self.guidance_scale, image_guidance_scale=self.image_guidance_scale
                ).images[0]
            elif (self.pix2pix_controlnet == "pix2pix"):
                feature = torch.cat(
                    [feature, feature, feature], dim=1)
                keyframe_image = self.pipe(
                    prompt,
                    image=self.control_image.resize((self.resolution,self.resolution)),
                    num_inference_steps=self.num_inference_steps,
                    image_guidance_scale=self.image_guidance_scale,
                    guidance_scale=self.guidance_scale,
                    class_labels=feature.float(),
                ).images[0]
            keyframe_gripper_image = keyframe_image
            self.current_goal_pil_img = keyframe_image
            if (self.steps_for_this_eps == 0):
                self.goal_img_list.append(np.asarray(
                    self.current_goal_pil_img.resize((200, 200))))
            self.image_goal = {"rgb_obs": {
                "rgb_static": "", "rgb_gripper": ""}}
            self.image_goal["rgb_obs"]["rgb_static"] = T.ToTensor()(np.array(
                keyframe_image.resize((200, 200)))).to(self.device).unsqueeze(0).unsqueeze(0) * 2 - 1
            self.image_goal["rgb_obs"]["rgb_gripper"] = T.ToTensor()(np.array(
                keyframe_gripper_image.resize((84, 84)))).to(self.device).unsqueeze(0).unsqueeze(0)*2 - 1
            # self.r3m_feature_2 = self.get_r3m_feature(keyframe_image)
            self.liv_feature_of_goal = self.get_liv_feature(
                T.functional.to_pil_image(self.image_goal["rgb_obs"]["rgb_static"].squeeze()))

        current_obs_jax = T.functional.to_pil_image(
            (obs["rgb_obs"]["rgb_static"].squeeze())).resize((200, 200))
        current_obs_jax = np.expand_dims((np.asarray(current_obs_jax)), axis=0)
        current_obs_jax = np.expand_dims(current_obs_jax, axis=0)
        goal_img_jax = (np.asarray(
            self.current_goal_pil_img.resize((200, 200))))
        goal_img_jax = np.expand_dims(goal_img_jax, axis=0)
        current_obs_jax = current_obs_jax
        goal_img_jax = goal_img_jax

        obs_input = {"image": current_obs_jax}
        goal_input = {"image": goal_img_jax}
        action = self.predict_action(obs_input, goal_input)

        return list(torch.tensor(action).squeeze())
