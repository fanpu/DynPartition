import numpy as np
import torch
import math
import random
import copy

def even_split(dependency_dict, parameter_dict, gpu_tolerance_dict):
	split_dict = {}
	total_params = 0
	allocator = {}
	for module_k in dependency_dict.keys():
		allocator[module_k] = []	
	for module_k in parameter_dict.keys():
		total_params += parameter_dict[module_k]
	
	total_gpus = len(gpu_tolerance_dict.keys())
	gpu_allocation = total_params / total_gpus 
	gpu_list = list(gpu_tolerance_dict.keys())

	param_tracer = 0
	split_counter = 0
	for moduel_k in dependency_dict.keys():
		dependency = dependency_dict[k]
		pred = dependency["pred"]
		dec = dependency["dec"]
		param = dependency["params"]
		if (param + param_tracer > gpu_tolerance_dict[gpu_list[split_counter]]):
			split_counter += 1
			split_dict[moduel_k] = split_counter
			gpu_name = gpu_list[split_counter]
			allocator[gpu_name].append(moduel_k)
			param_tracer = param
		else:
			split_dict[moduel_k] = split_counter
			gpu_name = gpu_list[split_counter]
			allocator[gpu_name].append(moduel_k)
			param_tracer += param
			if (param_tracer >= gpu_allocation):
				split_counter += 1
				param_tracer = 0
	if split_counter > total_gpus:
		print("insufficient number of gpus")
	return split_dict


def random_split(dependency_dict, gpu_tolerance_dict):
	total_gpus = len(gpu_tolerance_dict.keys())
	split_dict = {}
	gpu_allocated_dict = {}
	allocator = {}
	for module_k in dependency_dict.keys():
		allocator[module_k] = []	
	gpu_tolerance_dict = gpu_tolerance_dict.copy()
	for module_k in dependency_dict.keys():
		dependency = dependency_dict[module_k]
		valid_list = []
		params = dependency_dict[module_k]["params"]
		for gpu_k in gpu_tolerance_dict.keys():
			if gpu_tolerance_dict[gpu_k] > params:
				valid_list.append(gpu_k)
		gpu_idx = random.randint(0, len(valid_list))
		gpu_k = valid_list[gpu_idx]
		gpu_tolerance_dict[gpu_k] -= params
		split_dict[module_k] = gpu_k
		allocator[gpu_k].append(module_k)
	return allocator

def generate_dependency_dict(module_list):
	dependency_dict = {}
	for module in module_list:
		info_dict = {}
		name_key = module.name
		pred_names = module.pred
		dec_names = module.dec
		params_count = module.params
		info_dict["pred"] = pred_names
		info_dict["dec"] = dec_names
		info_dict["params"] = params_count
		dependency_dict[name_key] = info_dict
	return dependency_dict


def overlapping_dependency_dict(dependency_dict, gpu_tolerance_dict):
	total_gpus = len(gpu_tolerance_dict.keys())
	allocator = {}
	num_modules = len(dependency_dict.keys())	
	for module_k in dependency_dict.keys():
		allocator[module_k] = []
	for module_k in dependency_dict.keys():
		dependency = dependency_dict[module_k]
		valid_list = []
		params = dependency_dict[module_k]["params"]
		for gpu_k in gpu_tolerance_dict.keys():
			if gpu_tolerance_dict[gpu_k] > params:
				valid_list.append(gpu_k)
		gpu_idx = random.randint(0, len(valid_list))
		gpu_k = valid_list[gpu_idx]
		gpu_tolerance_dict[gpu_k] -= params
		allocator[gpu_k].append(module_k)
	for gpu_k in allocator.keys():
		allocation_status = {}

		searched_modules = 0
		for module_k in dependency_dict.keys():
			allocation_status[module_k] = False
		for module_in in allocator[gpu_k]:
			allocation_status[module_in] = True
			searched_modules += 1

		for module_tracer in allocator["gpu_k"]:
			dec_list = dependency_dict[module_tracer]["dec"]
			for dec in dec_list:
				if allocation_status[dec] == True:
					continue
				dec_param = dependency_dict[dec]["params"]
				if dec_param > gpu_tolerance_dict[gpu_k]:
					continue
				else:
					allocator[gpu_k].append(dec)
					allocation_status[dec] = True
					gpu_tolerance_dict -= dec_param
					searched_modules += 1
	return allocator
	
				




def push_to_target_gpus(module_dict, allocator):
	gpu_module_dict = {}
	for gpu_name in allocator.keys():
		gpu_module_dict[gpu_name] = {}
	for gpu_name in allocator.keys():
		device = gpu_name
		for module_k in allocator[gpu_name]:
			module_data = module_dict[module_k]["data"]
			cloned_module = copy.deepcopy(module_data)
			cloned_module.to(device)
			gpu_module_dict[device][module_k] = cloned_module


def iterate_through(gpu_module_dict, allocator, x_dict):
	module_k = x_dict["target_module"]
	device = x_dict["target_device"]
	data = x_dict["data"]
	data.to(device)
	module_data = gpu_module_dict[device][module_k]
	res = module_k(data)
	return res
	













