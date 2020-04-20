# bh_cases = []
# B = [0.46, 0,60, 0.72]
# H = [0.44, 0.60, 0.76]
# c2_h = 0.44
# for b in B:
#     for c0_h in H:
#         for c1_h in H:
#             for c3_h in H:
#                 for c4_h in H:
#                     for c5_h in H:
#                         for c6_h in H:
#                             # 7 类
#                             # ((b, h), (b, h), ..., (b, h))
#                             bh_cases.append(((b, c0_h), (b, c1_h), (b, c2_h), (b, c3_h), 
#                                           (b, c4_h), (b, c5_h), (b, c6_h)))



# results_file = open('results.txt', 'w')
# section_cases = []
# for bh_case in bh_cases:
#     c0_bh, c1_bh, c2_bh, c3_bh, c4_bh, c5_bh, c6_bh = bh_case
#     c0_b, c0_h = c0_bh
#     c1_b, c1_h = c1_bh
#     c2_b, c2_h = c2_bh
#     c3_b, c3_h = c3_bh
#     c4_b, c4_h = c4_bh
#     c5_b, c5_h = c5_bh
#     c6_b, c6_h = c6_bh
count = 0
c0_b = c1_b = c2_b = c3_b = c4_b = c5_b = c6_b = 0.46
c0_h = c1_h = c2_h = c3_h = c4_h = c5_h = c6_h = 0.76    
    # c0
for c0_t1 in np.arange(0.01, c0_b / 2, 0.01):
    c0_b2 = c0_b - 2 * c0_t1
    for c0_b1 in np.arange(0.1 * c0_t1 + 0.35, c0_h / 2, 0.2):
        c0_t2 = c0_h - 2 * c0_b1
        
        # c1
        for c1_t1 in np.arange(0.01, c1_b / 2, 0.01):
            c1_b2 = c1_b - 2 * c1_t1
            for c1_b1 in np.arange(0.1 * c1_t1 + 0.35, c1_h / 2, 0.2):
                c1_t2 = c1_h - 2 * c1_b1
                
                # c2
                for c2_t1 in np.arange(0.01, c2_b / 2, 0.01):
                    c2_b2 = c2_b - 2 * c2_t1
                    for c2_b1 in np.arange(0.1 * c2_t1 + 0.35, c2_h / 2, 0.2):
                        c2_t2 = c2_h - 2 * c2_b1
                        
                        # c3
                        for c3_t1 in np.arange(0.01, c3_b / 2, 0.01):
                            c3_b2 = c3_b - 2 * c3_t1
                            for c3_b1 in np.arange(0.1 * c3_t1 + 0.35, c3_h / 2, 0.2):
                                c3_t2 = c3_h - 2 * c3_b1
                                
                                # c4
                                for c4_t1 in np.arange(0.01, c4_b / 2, 0.01):
                                    c4_b2 = c4_b - 2 * c4_t1
                                    for c4_b1 in np.arange(0.1 * c4_t1 + 0.35, c4_h / 2, 0.2):
                                        c4_t2 = c4_h - 2 * c4_b1
                                        
                                        # c5
                                        for c5_t1 in np.arange(0.01, c5_b / 2, 0.01):
                                            c5_b2 = c5_b - 2 * c5_t1
                                            for c5_b1 in np.arange(0.1 * c5_t1 + 0.35, c5_h / 2, 0.2):
                                                c5_t2 = c5_h - 2 * c5_b1
                                                
                                                # c6
                                                for c6_t1 in np.arange(0.01, c6_b / 2, 0.01):
                                                    c6_b2 = c6_b - 2 * c6_t1
                                                    for c6_b1 in np.arange(0.1 * c6_t1 + 0.35, c6_h / 2, 0.2):
                                                        c6_t2 = c6_h - 2 * c6_b1
                                                        

                                                        count += 1
                                                        print(count)
                                                        # b2, t1, t2, b1
                                                        

                                                        # print((c0_b2, c0_t1, c0_t2, c0_b1)) 
                                                        # print((c1_b2, c1_t1, c1_t2, c1_b1))
                                                        # print((c2_b2, c2_t1, c2_t2, c2_b1)) 
                                                        # print((c3_b2, c3_t1, c3_t2, c3_b1)) 
                                                        # print((c4_b2, c4_t1, c4_t2, c4_b1)) 
                                                        # print((c5_b2, c5_t1, c5_t2, c5_b1)) 
                                                        # print((c6_b2, c6_t1, c6_t2, c6_b1))
                        
                                                        # bridge.check()

                                                        # beam_section_0 = [c0_b2, c0_t1, c0_t2, c0_b1]
                                                        # beam_section_1 = [c1_b2, c1_t1, c1_t2, c1_b1]
                                                        # beam_section_2 = [c2_b2, c2_t1, c2_t2, c2_b1]
                                                        # beam_section_3 = [c3_b2, c3_t1, c3_t2, c3_b1]
                                                        # beam_section_4 = [c4_b2, c4_t1, c4_t2, c4_b1]
                                                        # beam_section_5 = [c5_b2, c5_t1, c5_t2, c5_b1]
                                                        # beam_section_6 = [c6_b2, c6_t1, c6_t2, c6_b1]
                                                        # bridge.set_section_params(clusters[0], beam_section_0)
                                                        # bridge.set_section_params(clusters[1], beam_section_1)
                                                        # bridge.set_section_params(clusters[2], beam_section_2)
                                                        # bridge.set_section_params(clusters[3], beam_section_3)
                                                        # bridge.set_section_params(clusters[4], beam_section_4)
                                                        # bridge.set_section_params(clusters[5], beam_section_5)
                                                        # bridge.set_section_params(clusters[6], beam_section_6)
                                                                        
                                                        # bridge.update()

                                                        

                                                        
                                                        # qualified = bridge.stiffness_check()
                                                        # if qualified:
                                                        #     current_case = (
                                                        #         (c0_b2, c0_t1, c0_t2, c0_b1), 
                                                        #         (c1_b2, c1_t1, c1_t2, c1_b1), 
                                                        #         (c2_b2, c2_t1, c2_t2, c2_b1), 
                                                        #         (c3_b2, c3_t1, c3_t2, c3_b1), 
                                                        #         (c4_b2, c4_t1, c4_t2, c4_b1), 
                                                        #         (c5_b2, c5_t1, c5_t2, c5_b1), 
                                                        #         (c6_b2, c6_t1, c6_t2, c6_b1)
                                                        #     )
                                                        #     print('合格\n')
                                                        #     print(current_case, file=results_file)
                                                        # else:
                                                        #     print('不合格\n')
                                
                                
        
# results_file.close()





