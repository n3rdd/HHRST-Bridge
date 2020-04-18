

class Check:
	def __init__(self, bridge):
		self.bridge = bridge


	def set_section_params(self):
		assert bridge.checking, '当前不在检算状态.'

        for unit_num in units_nums_group:
            unit = self.units[unit_num]
            unit.beam_section_data = beam_section_data

	def fatigue_check(self):
		pass

	def strength_check(self, unit, params):
		pass

	def stiffness_check(self, unit, params):
		pass

	def overall_stability_check(self, unit, params):
		pass

	def local_stability_check(self, unit, params):
		pass



