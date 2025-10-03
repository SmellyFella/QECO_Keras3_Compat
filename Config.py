for time_index in range(Config.N_TIME_SLOT):
            for meter_index in range(Config.N_METER):
                action = meter_RL_list[meter_index].choose_action(state)
                action_counts[action] += 1

                reward, breakdown = QoE_Function(...)
                
                # log breakdown
                for k, v in breakdown.items():
                    reward_components[k].append(v)
